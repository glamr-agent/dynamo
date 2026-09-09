/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// These tests render the production sidecarScriptTemplate through renderSidecarScript
// and actually execute the result, because the defect they guard is a runtime
// behavior: the script used to loop forever when kubectl was unusable. Asserting on
// the text of the constant would pass with the fix reverted and prove nothing.
//
// None of these tests may call t.Parallel: the rendered script writes to the fixed
// paths /tmp/progress.yaml and /tmp/cm.yaml, which are container-local in production
// but shared on a test host.

// posixUtilitiesUsedByScript lists the external binaries the rendered script invokes,
// excluding kubectl. Each test builds a PATH containing exactly these, which is what
// lets a test decide whether kubectl is present.
var posixUtilitiesUsedByScript = []string{"date", "grep", "awk", "sed", "tr", "cat", "sleep"}

// pipefailShell resolves a shell that accepts "set -o pipefail", which the rendered
// script requires on its second line. The production sidecar image satisfies this
// (bitnami/kubectl's /bin/sh is bash, and BusyBox ash supports the option), but a
// Debian-family test host points /bin/sh at dash, which rejects it.
func pipefailShell(t *testing.T) string {
	t.Helper()

	for _, candidate := range []string{"sh", "bash", "ash"} {
		path, err := exec.LookPath(candidate)
		if err != nil {
			continue
		}
		if err := exec.Command(path, "-c", "set -o pipefail").Run(); err == nil {
			return path
		}
	}

	t.Skip("no shell supporting 'set -o pipefail' is available on this host")

	return ""
}

// isolatedPATHDir builds a directory of symlinks to the POSIX utilities the script
// uses and returns it. A test sets PATH to this directory alone, so the only way
// kubectl can be resolved is if the test puts a shim in it.
func isolatedPATHDir(t *testing.T) string {
	t.Helper()

	dir := t.TempDir()
	for _, utility := range posixUtilitiesUsedByScript {
		resolved, err := exec.LookPath(utility)
		if err != nil {
			t.Skipf("the script requires %q, which is not available on this host: %v", utility, err)
		}
		if err := os.Symlink(resolved, filepath.Join(dir, utility)); err != nil {
			t.Fatalf("failed to link %q into the isolated PATH: %v", utility, err)
		}
	}

	return dir
}

// writeKubectlShim installs a fake kubectl in pathDir that appends its own argv to
// logFile and then exits with exitCode. When exitCode is zero, a "get" invocation
// answers with a payload containing "terminated", which is what the script's poll
// loop greps for.
func writeKubectlShim(t *testing.T, pathDir string, logFile string, exitCode int) {
	t.Helper()

	shim := `#!/bin/sh
echo "$@" >> ` + logFile + `
if [ "$1" = "get" ]; then
  echo '{"terminated":{"exitCode":0,"reason":"Completed"}}'
fi
exit ` + strconv.Itoa(exitCode) + `
`
	if err := os.WriteFile(filepath.Join(pathDir, "kubectl"), []byte(shim), 0o755); err != nil {
		t.Fatalf("failed to install the kubectl shim: %v", err)
	}
}

// seedProfilerOutput writes the files the profiler would have left in the output
// volume: a terminal status file and the generated DGD config the sidecar copies.
func seedProfilerOutput(t *testing.T, outputDir string, status string) {
	t.Helper()

	statusFile := "status: " + status + "\nphase: Done\nmessage: profiling finished\n"
	if err := os.WriteFile(filepath.Join(outputDir, "profiler_status.yaml"), []byte(statusFile), 0o644); err != nil {
		t.Fatalf("failed to seed profiler_status.yaml: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, ProfilingOutputFile), []byte("spec:\n  services: {}\n"), 0o644); err != nil {
		t.Fatalf("failed to seed %s: %v", ProfilingOutputFile, err)
	}
}

// renderOutputCopierScript renders the production template with the same key set the
// job builder uses, differing only in the output path and the failure deadline.
func renderOutputCopierScript(t *testing.T, outputDir string, kubectlFailureDeadlineSeconds int) string {
	t.Helper()

	script, err := renderSidecarScript(map[string]string{
		"OutputPath":                    outputDir,
		"OutputFile":                    ProfilingOutputFile,
		"ConfigMapName":                 "dgdr-output-test",
		"Namespace":                     "test-namespace",
		"DGDRName":                      "test-dgdr",
		"DGDRuid":                       "11111111-2222-3333-4444-555555555555",
		"KubectlFailureDeadlineSeconds": strconv.Itoa(kubectlFailureDeadlineSeconds),
	})
	if err != nil {
		t.Fatalf("failed to render the sidecar script: %v", err)
	}

	return script
}

// runOutputCopierScript executes the rendered script with PATH set to pathDir alone
// and returns its exit code, combined output, and how long it ran. An exit code of
// -1 means the script was still running when the bound elapsed, which is the hang
// this file exists to detect.
func runOutputCopierScript(t *testing.T, script string, pathDir string, bound time.Duration) (int, string, time.Duration) {
	t.Helper()

	shell := pipefailShell(t)
	ctx, cancel := context.WithTimeout(context.Background(), bound)
	defer cancel()

	cmd := exec.CommandContext(ctx, shell, "-c", script)
	cmd.Env = []string{"PATH=" + pathDir, "HOSTNAME=profiling-job-test-pod"}

	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output

	// Without WaitDelay a killed script leaves its "sleep" child holding the output
	// pipe open, and Wait would block past the bound instead of reporting the hang.
	cmd.WaitDelay = 5 * time.Second

	start := time.Now()
	err := cmd.Run()
	elapsed := time.Since(start)

	if ctx.Err() != nil {
		return -1, output.String(), elapsed
	}
	if err == nil {
		return 0, output.String(), elapsed
	}

	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode(), output.String(), elapsed
	}

	t.Fatalf("failed to run the sidecar script: %v", err)

	return 0, "", elapsed
}

// TestOutputCopierScriptFailsFastWhenKubectlIsMissing is the regression barrier. With
// the preflight reverted the script spins on "sleep 10" forever, so this test hits
// the bound and fails, reproducing the reported hang.
func TestOutputCopierScriptFailsFastWhenKubectlIsMissing(t *testing.T) {
	outputDir := t.TempDir()
	pathDir := isolatedPATHDir(t)

	t.Log("Render the production sidecar script with the production failure deadline")
	script := renderOutputCopierScript(t, outputDir, outputCopierKubectlFailureDeadlineSeconds)

	t.Log("Execute it on a PATH that has every utility the script needs except kubectl")
	exitCode, output, elapsed := runOutputCopierScript(t, script, pathDir, 30*time.Second)
	t.Logf("script exited with %d after %s", exitCode, elapsed)

	t.Log("The sidecar must terminate non-zero so the Job can reach JobFailed")
	if exitCode == -1 {
		t.Fatalf("the sidecar never terminated without kubectl; this is the reported hang. output:\n%s", output)
	}
	if exitCode == 0 {
		t.Fatalf("the sidecar exited 0 without kubectl, which the controller reads as a recorded profiler failure. output:\n%s", output)
	}

	t.Log("The diagnostic must name kubectl, because this container's log is the only place a user can read why it gave up")
	if !strings.Contains(output, "kubectl") {
		t.Errorf("the sidecar exited without naming kubectl in its output:\n%s", output)
	}
}

// TestOutputCopierScriptCompletesWhenKubectlWorks is the negative control: the normal
// path must be untouched by the fix. It passes both before and after the change.
func TestOutputCopierScriptCompletesWhenKubectlWorks(t *testing.T) {
	outputDir := t.TempDir()
	pathDir := isolatedPATHDir(t)
	shimLog := filepath.Join(t.TempDir(), "kubectl.log")

	t.Log("Install a kubectl shim that reports the profiler as terminated and accepts applies")
	writeKubectlShim(t, pathDir, shimLog, 0)

	t.Log("Seed the output volume with a successful profiler run")
	seedProfilerOutput(t, outputDir, "success")

	t.Log("Execute the rendered production script")
	script := renderOutputCopierScript(t, outputDir, outputCopierKubectlFailureDeadlineSeconds)
	exitCode, output, elapsed := runOutputCopierScript(t, script, pathDir, 60*time.Second)
	t.Logf("script exited with %d after %s", exitCode, elapsed)

	if exitCode != 0 {
		t.Fatalf("the sidecar failed on the healthy path, exit %d. output:\n%s", exitCode, output)
	}

	t.Log("The results must still be written back to the ConfigMap")
	recorded, err := os.ReadFile(shimLog)
	if err != nil {
		t.Fatalf("failed to read the kubectl shim log: %v", err)
	}
	if !strings.Contains(string(recorded), "apply") {
		t.Errorf("the sidecar never applied the output ConfigMap. kubectl invocations:\n%s", recorded)
	}
}

// TestOutputCopierScriptExitsWhenKubectlAlwaysFails covers the case the preflight
// cannot catch: kubectl is installed but every call fails, as with revoked RBAC or an
// unreachable API server. The first failure must be tolerated as a transient and the
// sustained failure must terminate the container.
func TestOutputCopierScriptExitsWhenKubectlAlwaysFails(t *testing.T) {
	outputDir := t.TempDir()
	pathDir := isolatedPATHDir(t)
	shimLog := filepath.Join(t.TempDir(), "kubectl.log")

	t.Log("Install a kubectl shim that fails every invocation")
	writeKubectlShim(t, pathDir, shimLog, 1)

	t.Log("Seed a still-running profiler so the loop keeps relaying progress")
	seedProfilerOutput(t, outputDir, "running")

	t.Log("Render with a one-second deadline so the test does not wait out the production five minutes")
	script := renderOutputCopierScript(t, outputDir, 1)

	t.Log("Execute it; the loop polls every ten seconds, so the deadline trips on the second pass")
	exitCode, output, elapsed := runOutputCopierScript(t, script, pathDir, 90*time.Second)
	t.Logf("script exited with %d after %s", exitCode, elapsed)

	if exitCode == -1 {
		t.Fatalf("the sidecar never terminated while kubectl failed continuously. output:\n%s", output)
	}
	if exitCode == 0 {
		t.Fatalf("the sidecar exited 0 while kubectl failed continuously, which the controller reads as a recorded profiler failure. output:\n%s", output)
	}

	t.Log("The first failure must have been tolerated rather than terminating on contact")
	recorded, err := os.ReadFile(shimLog)
	if err != nil {
		t.Fatalf("failed to read the kubectl shim log: %v", err)
	}
	if attempts := strings.Count(strings.TrimSpace(string(recorded)), "\n") + 1; attempts < 2 {
		t.Errorf("expected the sidecar to retry kubectl before giving up, got %d invocation(s):\n%s", attempts, recorded)
	}
}

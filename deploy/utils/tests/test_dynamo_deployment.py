# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``DynamoDeploymentClient``.

Fail-fast behavior added for #9213: when a candidate deployment's worker
pods enter ``CrashLoopBackOff``,
``DynamoDeploymentClient.wait_for_deployment_ready`` must raise
``DeploymentFailedError`` immediately rather than waiting out the full
``timeout`` — otherwise the thorough-mode profiler burns up to 30 min
of wall-clock per failing candidate.

Schema handling: ``create_deployment`` is handed both DynamoGraphDeployment
schemas — the profiler emits v1beta1 (``spec.components`` as a list), while
the CLI and older manifests still use v1alpha1 (``spec.services`` as a
mapping). It must read the component names out of either shape and address
the CRD version the manifest declares.

The v1beta1 case is driven through ``create_deployment`` because that is the
path that used to raise ``KeyError: 'services'``. The v1alpha1 and
empty-manifest cases are driven through ``extract_component_names`` and
``detect_dgd_crd_version`` directly — see the comment above them.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip the whole module if the deploy.utils runtime deps aren't available
# in this environment. The test doesn't actually exercise these — it just
# needs the import of `dynamo_deployment` to succeed.
pytest.importorskip("aiofiles")
pytest.importorskip("kubernetes_asyncio")
pytest.importorskip("httpx")

from deploy.utils.dynamo_deployment import (  # noqa: E402
    DeploymentFailedError,
    DynamoDeploymentClient,
    detect_dgd_crd_version,
    extract_component_names,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


async def test_wait_for_deployment_ready_raises_deployment_failed_on_crashloop(
    monkeypatch,
):
    client = DynamoDeploymentClient(namespace="ns", deployment_name="dgd-test")
    client.deployment_name = "dgd-test"
    client._original_components = ["PrefillWorker"]
    client.components = ["prefillworker"]

    # DGD CR exists but isn't Ready yet.
    client.custom_api = MagicMock()
    client.custom_api.get_namespaced_custom_object = AsyncMock(
        return_value={"status": {"state": "deploying", "conditions": []}}
    )
    # Simulate a crash on the very first poll.
    client._detect_terminal_pod_failure = AsyncMock(  # type: ignore[method-assign]
        return_value="pod p0 container worker in CrashLoopBackOff"
    )

    # Avoid sleeping in the test.
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("deploy.utils.dynamo_deployment.asyncio.sleep", _no_sleep)

    with pytest.raises(DeploymentFailedError) as excinfo:
        # Pass a generous timeout so a regression (timeout instead of
        # raise) would be obvious.
        await client.wait_for_deployment_ready(timeout=600)

    assert "CrashLoopBackOff" in str(excinfo.value)
    # Confirm we didn't run out the timeout — there should have been at
    # most one DGD status check before the raise.
    assert client.custom_api.get_namespaced_custom_object.await_count == 1


def _client_with_mocked_api(deployment_name: str) -> DynamoDeploymentClient:
    """A client whose Kubernetes setup and API object are mocked out.

    ``create_deployment`` calls ``_init_kubernetes`` first, so stubbing it
    leaves the ``custom_api`` assigned here in place and the real parsing
    path is still the one under test.
    """
    client = DynamoDeploymentClient(namespace="ns", deployment_name=deployment_name)
    client._init_kubernetes = AsyncMock()  # type: ignore[method-assign]
    client.custom_api = MagicMock()
    client.custom_api.create_namespaced_custom_object = AsyncMock()
    return client


async def test_create_deployment_reads_v1beta1_components_list():
    client = _client_with_mocked_api("dgd-v1beta1")
    spec = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "dgd-v1beta1"},
        "spec": {
            "components": [
                {"name": "Frontend", "replicas": 1},
                {"name": "VllmPrefillWorker", "replicas": 2},
            ]
        },
    }

    await client.create_deployment(spec)

    # These names are stamped onto the `nvidia.com/dynamo-component` pod
    # label the log fetch and the terminal-failure check select on.
    assert client._original_components == ["Frontend", "VllmPrefillWorker"]
    # The lowercase derivative is what the log directory layout and
    # `pick_decode_component` consume.
    assert client.components == ["frontend", "vllmprefillworker"]

    _, kwargs = client.custom_api.create_namespaced_custom_object.await_args
    # The request path has to address the same version the body declares,
    # or the API server rejects the create.
    assert kwargs["version"] == "v1beta1"


# The v1alpha1 cases below call the helpers directly; driving them through
# `create_deployment` would pin no branch that these helpers introduced.


def test_extract_component_names_reads_v1alpha1_services_mapping():
    names = extract_component_names(
        {
            "spec": {
                "services": {
                    "Frontend": {"replicas": 1},
                    "VllmPrefillWorker": {"replicas": 2},
                }
            }
        },
        "dgd-v1alpha1",
    )

    assert names == ["Frontend", "VllmPrefillWorker"]


def test_detect_dgd_crd_version_trusts_api_version():
    assert detect_dgd_crd_version({"apiVersion": "nvidia.com/v1alpha1"}) == "v1alpha1"
    assert detect_dgd_crd_version({"apiVersion": "nvidia.com/v1beta1"}) == "v1beta1"


def test_detect_dgd_crd_version_falls_back_to_spec_shape():
    # Unlabelled manifests are real: `main()` accepts an arbitrary YAML file,
    # so the spec shape has to decide when `apiVersion` is absent.
    assert detect_dgd_crd_version({"spec": {"components": []}}) == "v1beta1"
    assert detect_dgd_crd_version({"spec": {"services": {}}}) == "v1alpha1"


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param({"components": []}, id="empty-components-list"),
        pytest.param({"components": [{"replicas": 1}]}, id="components-without-name"),
        pytest.param({"services": {}}, id="empty-services-mapping"),
        pytest.param({}, id="neither-field"),
    ],
)
def test_extract_component_names_raises_when_no_names_resolve(spec):
    # Returning [] instead of raising would silently disable the CrashLoopBackOff
    # fail-fast, which skips out on an empty `_original_components`.
    with pytest.raises(ValueError) as excinfo:
        extract_component_names({"spec": spec}, "dgd-empty")

    assert "dgd-empty" in str(excinfo.value)

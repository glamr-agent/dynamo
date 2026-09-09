/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"context"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// TestGroveRenderPreservesTopologyConstraintOnReplicaChange renders an existing
// constrained PodCliqueSet again after only a component replica count changed,
// which is the shape a `replicas` JSON patch against a running
// DynamoGraphDeployment produces. Horizontal changes reach Grove through the
// generated child's scale subresource, so the re-render must leave both the
// recorded topology constraint and the PodCliqueSet template replicas alone.
func TestGroveRenderPreservesTopologyConstraintOnReplicaChange(t *testing.T) {
	ctx := context.Background()
	tests := map[string]struct {
		deployment                  *v1beta1.DynamoGraphDeployment
		scaledComponent             string
		scaledReplicas              int32
		wantTemplateConstraint      *grovev1alpha1.TopologyConstraint
		wantCliqueReplicas          map[string]int32
		wantScalingGroupConstraints map[string]*grovev1alpha1.TopologyConstraint
		wantScalingGroupReplicas    map[string]int32
	}{
		"standalone clique keeps the deployment-level pack domain": {
			deployment: &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "vllm-agg-tas", Namespace: "default"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					TopologyConstraint: &v1beta1.SpecTopologyConstraint{
						ClusterTopologyName: "default",
						PackDomain:          v1beta1.TopologyDomain("rack"),
					},
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: "Frontend",
							ComponentType: v1beta1.ComponentTypeFrontend,
							Replicas:      ptr.To(int32(1)),
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{{Name: "main", Image: "vllm-runtime:test"}},
								},
							},
						},
						{
							ComponentName: "VllmDecodeWorker",
							ComponentType: v1beta1.ComponentTypeWorker,
							Replicas:      ptr.To(int32(1)),
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{{Name: "main", Image: "vllm-runtime:test"}},
								},
							},
						},
					},
				},
			},
			scaledComponent: "VllmDecodeWorker",
			scaledReplicas:  2,
			wantTemplateConstraint: &grovev1alpha1.TopologyConstraint{
				TopologyName: "default",
				Pack: &grovev1alpha1.TopologyPackConstraint{
					RequiredDomain: grovev1alpha1.TopologyDomain("rack"),
				},
			},
			wantCliqueReplicas: map[string]int32{"frontend": 1, "vllmdecodeworker": 1},
		},
		"scaling group keeps its component-level pack domain": {
			deployment: &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "vllm-agg-tas-multinode", Namespace: "default"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					TopologyConstraint: &v1beta1.SpecTopologyConstraint{
						ClusterTopologyName: "default",
						PackDomain:          v1beta1.TopologyDomain("zone"),
					},
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: "Frontend",
							ComponentType: v1beta1.ComponentTypeFrontend,
							Replicas:      ptr.To(int32(1)),
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{{Name: "main", Image: "vllm-runtime:test"}},
								},
							},
						},
						{
							ComponentName: "VllmDecodeWorker",
							ComponentType: v1beta1.ComponentTypeWorker,
							Replicas:      ptr.To(int32(1)),
							Multinode:     &v1beta1.MultinodeSpec{NodeCount: 2},
							TopologyConstraint: &v1beta1.TopologyConstraint{
								PackDomain: v1beta1.TopologyDomain("rack"),
							},
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{{Name: "main", Image: "vllm-runtime:test"}},
								},
							},
						},
					},
				},
			},
			scaledComponent: "VllmDecodeWorker",
			scaledReplicas:  2,
			wantTemplateConstraint: &grovev1alpha1.TopologyConstraint{
				TopologyName: "default",
				Pack: &grovev1alpha1.TopologyPackConstraint{
					RequiredDomain: grovev1alpha1.TopologyDomain("zone"),
				},
			},
			wantScalingGroupConstraints: map[string]*grovev1alpha1.TopologyConstraint{
				"vllmdecodeworker": {
					Pack: &grovev1alpha1.TopologyPackConstraint{
						RequiredDomain: grovev1alpha1.TopologyDomain("rack"),
					},
				},
			},
			wantScalingGroupReplicas: map[string]int32{"vllmdecodeworker": 1},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Log("render the PodCliqueSet the operator creates for the constrained deployment")
			createRenderer := newGroveWorkloadRenderer(
				fake.NewClientBuilder().
					WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
					WithObjects(tt.deployment.DeepCopy()).
					Build(),
				&configv1alpha1.OperatorConfiguration{},
				&controller_common.RuntimeConfig{},
				nil,
			)
			created, err := createRenderer.Render(ctx, tt.deployment.DeepCopy(), nil, nil, false)
			require.NoError(t, err)
			require.Equal(
				t,
				tt.wantTemplateConstraint,
				created.desired.Spec.Template.TopologyConstraint,
				"creation path must record the deployment-level topology constraint",
			)

			t.Log("patch only the component replica count on the existing deployment")
			scaled := tt.deployment.DeepCopy()
			scaledComponent := scaled.GetComponentByName(tt.scaledComponent)
			require.NotNil(t, scaledComponent, "component %s", tt.scaledComponent)
			require.NotEqual(t, tt.scaledReplicas, *scaledComponent.Replicas, "the patch must change the replica count")
			scaledComponent.Replicas = ptr.To(tt.scaledReplicas)

			t.Log("re-render against the PodCliqueSet already running in the cluster")
			scaleRenderer := newGroveWorkloadRenderer(
				fake.NewClientBuilder().
					WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
					WithObjects(scaled.DeepCopy(), created.desired.DeepCopy()).
					Build(),
				&configv1alpha1.OperatorConfiguration{},
				&controller_common.RuntimeConfig{},
				nil,
			)
			rendered, err := scaleRenderer.Render(ctx, scaled, nil, nil, false)
			require.NoError(t, err)
			desired := rendered.desired

			t.Log("assert the replica change leaves the recorded topology constraint untouched")
			require.Equal(
				t,
				tt.wantTemplateConstraint,
				desired.Spec.Template.TopologyConstraint,
				"replica change must not drop or rewrite the PodCliqueSet topology constraint",
			)
			require.Equal(
				t,
				created.desired.Spec.Template.TopologyConstraint,
				desired.Spec.Template.TopologyConstraint,
				"the live constraint must stay the one recorded at creation",
			)
			for scalingGroupName, wantConstraint := range tt.wantScalingGroupConstraints {
				config := requireGroveScalingGroupConfig(t, desired, scalingGroupName)
				require.Equal(
					t,
					wantConstraint,
					config.TopologyConstraint,
					"scaling group %s topology constraint",
					scalingGroupName,
				)
			}

			t.Log("assert the replica change is not routed through the PodCliqueSet spec")
			for cliqueName, wantReplicas := range tt.wantCliqueReplicas {
				clique := requireGroveClique(t, desired, cliqueName)
				require.Equal(t, wantReplicas, clique.Spec.Replicas, "clique %s replicas", cliqueName)
			}
			for scalingGroupName, wantReplicas := range tt.wantScalingGroupReplicas {
				config := requireGroveScalingGroupConfig(t, desired, scalingGroupName)
				require.NotNil(t, config.Replicas, "scaling group %s replicas", scalingGroupName)
				require.Equal(t, wantReplicas, *config.Replicas, "scaling group %s replicas", scalingGroupName)
			}
		})
	}
}

func requireGroveScalingGroupConfig(
	t *testing.T,
	pcs *grovev1alpha1.PodCliqueSet,
	name string,
) *grovev1alpha1.PodCliqueScalingGroupConfig {
	t.Helper()
	for i := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		config := &pcs.Spec.Template.PodCliqueScalingGroupConfigs[i]
		if config.Name == name {
			return config
		}
	}
	t.Fatalf("expected rendered grove scaling group config %q", name)
	return nil
}

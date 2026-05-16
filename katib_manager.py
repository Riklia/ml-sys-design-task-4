"""
Kubeflow Katib Hyperparameter Tuning Integration
Direct Katib invocation and management through KFP.

This module provides:
1. Standalone trial worker (runs as Katib trial)
2. Direct Katib experiment creation and submission
3. Trial monitoring and results collection
"""

import os
import json
import yaml
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from kubernetes import client, config
from kubeflow.katib import V1beta1Experiment, V1beta1AlgorithmSpec, V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec, V1beta1NasConfig, V1beta1TrialTemplate
from kubeflow.katib import V1beta1TrialSpec
from kubeflow.katib import ApiClient, CustomObjectsApi
import time


@dataclass
class KatibParameter:
    """Katib hyperparameter definition."""
    name: str
    parameterType: str  # double, int, discrete, categorical
    feasibleSpace: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class KatibObjective:
    """Katib optimization objective."""
    type: str  # maximize or minimize
    goal: Optional[float] = None
    objectiveMetricName: str = "val_accuracy"
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class KatibExperimentManager:
    """
    Manages Katib experiments creation, submission, and monitoring.
    
    Example:
        manager = KatibExperimentManager(namespace='kubeflow')
        
        # Define parameters
        params = [
            KatibParameter('learning_rate', 'double', {'min': '0.0001', 'max': '0.01'}),
            KatibParameter('batch_size', 'categorical', {'list': ['32', '64', '128']}),
        ]
        
        # Create experiment
        exp = manager.create_experiment(
            name='my-tuning',
            parameters=params,
            trial_count=9,
            parallel_trials=3,
            algorithm='bayesianoptimization'
        )
        
        # Monitor
        manager.watch_experiment('my-tuning')
        
        # Get results
        best = manager.get_best_parameters('my-tuning')
    """
    
    def __init__(self, namespace: str = 'kubeflow', 
                 kubeflow_namespace: str = 'kubeflow'):
        """
        Initialize Katib manager.
        
        Args:
            namespace: K8s namespace for experiments
            kubeflow_namespace: Kubeflow system namespace
        """
        self.namespace = namespace
        self.kubeflow_namespace = kubeflow_namespace
        
        # Load kubeconfig
        try:
            config.load_kube_config()
        except Exception:
            try:
                config.load_incluster_config()
            except:
                print("[KATIB] Warning: Could not load kubeconfig")
        
        self.api_client = client.ApiClient()
        self.custom_api = CustomObjectsApi(self.api_client)
        self.v1_client = client.CoreV1Api()
        
        print(f"[KATIB] Manager initialized for namespace: {namespace}")
    
    def create_experiment(
        self,
        name: str,
        parameters: List[KatibParameter],
        trial_count: int = 9,
        parallel_trials: int = 3,
        algorithm: str = 'random',
        objective: Optional[KatibObjective] = None,
        trial_template_yaml: Optional[str] = None,
        max_trial_count: Optional[int] = None,
        nas_config: Optional[Dict] = None,
    ) -> V1beta1Experiment:
        """
        Create a Katib experiment definition.
        
        Args:
            name: Experiment name
            parameters: List of KatibParameter objects
            trial_count: Total number of trials (deprecated, use max_trial_count)
            parallel_trials: Concurrent trials
            algorithm: Optimization algorithm (random, grid, bayesianoptimization, hyperband, tpe)
            objective: Optimization objective (default: maximize val_accuracy)
            trial_template_yaml: Custom trial template YAML
            max_trial_count: Maximum trials (replaces trial_count)
            nas_config: NAS configuration if using NAS algorithm
        
        Returns:
            V1beta1Experiment object
        """
        print(f"\n[KATIB] Creating experiment: {name}")
        print(f"[KATIB] Algorithm: {algorithm}")
        print(f"[KATIB] Trial count: {trial_count}")
        print(f"[KATIB] Parallel trials: {parallel_trials}")
        
        # Use max_trial_count if provided
        max_trials = max_trial_count or trial_count
        
        # Default objective
        if objective is None:
            objective = KatibObjective(
                type='maximize',
                objectiveMetricName='val_accuracy',
                goal=0.99
            )
        
        # Algorithm spec
        algorithm_spec = {
            'algorithmName': algorithm,
        }
        
        # Add algorithm-specific settings
        if algorithm == 'bayesianoptimization':
            algorithm_spec['algorithmSettings'] = [
                {'name': 'random_state', 'value': '10'}
            ]
        elif algorithm == 'hyperband':
            algorithm_spec['algorithmSettings'] = [
                {'name': 'eta', 'value': '3'},
                {'name': 'min_resource', 'value': '1'},
                {'name': 'max_resource', 'value': '81'},
                {'name': 'r', 'value': '9'}
            ]
        
        # Convert parameters to dict format
        param_specs = []
        for param in parameters:
            spec = {
                'name': param.name,
                'parameterType': param.parameterType,
                'feasibleSpace': param.feasibleSpace
            }
            param_specs.append(spec)
            print(f"[KATIB] Parameter: {param.name} ({param.parameterType})")
        
        # Trial template - default KFP pipeline
        if trial_template_yaml is None:
            trial_template_yaml = self._get_default_trial_template()
        
        # Create experiment spec
        experiment_spec = {
            'apiVersion': 'kubeflow.org/v1beta1',
            'kind': 'Experiment',
            'metadata': {
                'name': name,
                'namespace': self.namespace,
            },
            'spec': {
                'algorithm': algorithm_spec,
                'parallelTrialCount': parallel_trials,
                'maxTrialCount': max_trials,
                'objective': objective.to_dict(),
                'parameters': param_specs,
                'trialTemplate': trial_template_yaml,
            }
        }
        
        # Add NAS config if provided
        if nas_config:
            experiment_spec['spec']['nasConfig'] = nas_config
        
        print(f"[KATIB] Experiment spec created")
        
        return experiment_spec
    
    def submit_experiment(self, experiment_spec: Dict) -> bool:
        """
        Submit experiment to Katib.
        
        Args:
            experiment_spec: Experiment specification dict
        
        Returns:
            True if successful
        """
        try:
            name = experiment_spec['metadata']['name']
            print(f"\n[KATIB] Submitting experiment: {name}")
            
            # Create experiment via custom resource
            self.custom_api.create_namespaced_custom_object(
                group='kubeflow.org',
                version='v1beta1',
                namespace=self.namespace,
                plural='experiments',
                body=experiment_spec
            )
            
            print(f"[KATIB] ✅ Experiment submitted successfully")
            return True
            
        except Exception as e:
            print(f"[KATIB] ❌ Failed to submit experiment: {str(e)}")
            return False
    
    def create_and_submit_experiment(
        self,
        name: str,
        parameters: List[KatibParameter],
        **kwargs
    ) -> bool:
        """
        Create and submit experiment in one call.
        
        Args:
            name: Experiment name
            parameters: List of KatibParameter objects
            **kwargs: Additional arguments for create_experiment
        
        Returns:
            True if successful
        """
        spec = self.create_experiment(name, parameters, **kwargs)
        return self.submit_experiment(spec)
    
    def get_experiment(self, name: str) -> Optional[Dict]:
        """Get experiment status."""
        try:
            exp = self.custom_api.get_namespaced_custom_object(
                group='kubeflow.org',
                version='v1beta1',
                namespace=self.namespace,
                plural='experiments',
                name=name
            )
            return exp
        except Exception as e:
            print(f"[KATIB] Error getting experiment: {str(e)}")
            return None
    
    def list_experiments(self) -> List[str]:
        """List all experiments in namespace."""
        try:
            exps = self.custom_api.list_namespaced_custom_object(
                group='kubeflow.org',
                version='v1beta1',
                namespace=self.namespace,
                plural='experiments'
            )
            return [exp['metadata']['name'] for exp in exps.get('items', [])]
        except Exception as e:
            print(f"[KATIB] Error listing experiments: {str(e)}")
            return []
    
    def get_trials(self, experiment_name: str) -> List[Dict]:
        """Get all trials for an experiment."""
        try:
            trials = self.custom_api.list_namespaced_custom_object(
                group='kubeflow.org',
                version='v1beta1',
                namespace=self.namespace,
                plural='trials',
                label_selector=f'exp={experiment_name}'
            )
            return trials.get('items', [])
        except Exception as e:
            print(f"[KATIB] Error getting trials: {str(e)}")
            return []
    
    def get_best_parameters(self, experiment_name: str) -> Optional[Dict]:
        """Get best parameters found by Katib."""
        exp = self.get_experiment(experiment_name)
        if not exp:
            return None
        
        try:
            status = exp.get('status', {})
            best_params = status.get('bestTrialParameters', [])
            best_metric = status.get('bestTrialValue')
            
            # Convert list format to dict
            params_dict = {}
            for param in best_params:
                params_dict[param['name']] = param['value']
            
            result = {
                'parameters': params_dict,
                'best_metric': best_metric,
                'best_trial': status.get('bestTrial', {}),
                'completion_time': status.get('completionTime')
            }
            
            print(f"\n[KATIB] Best parameters found:")
            for key, value in params_dict.items():
                print(f"  {key}: {value}")
            print(f"  Best metric: {best_metric}")
            
            return result
            
        except Exception as e:
            print(f"[KATIB] Error extracting best parameters: {str(e)}")
            return None
    
    def watch_experiment(self, experiment_name: str, timeout: int = 3600) -> bool:
        """
        Watch experiment until completion.
        
        Args:
            experiment_name: Name of experiment to watch
            timeout: Maximum time to wait (seconds)
        
        Returns:
            True if completed successfully
        """
        print(f"\n[KATIB] Watching experiment: {experiment_name}")
        start_time = time.time()
        
        while True:
            exp = self.get_experiment(experiment_name)
            if not exp:
                print(f"[KATIB] ❌ Experiment not found")
                return False
            
            status = exp.get('status', {})
            phase = status.get('phase', 'Unknown')
            trials_run = status.get('trialsSucceeded', 0)
            trials_total = exp.get('spec', {}).get('maxTrialCount', '?')
            
            print(f"\r[KATIB] Phase: {phase} | Trials: {trials_run}/{trials_total}", end='', flush=True)
            
            # Check completion
            if phase in ['Succeeded', 'Failed', 'Stopped']:
                print()  # New line
                print(f"[KATIB] ✅ Experiment {phase}")
                return phase == 'Succeeded'
            
            # Check timeout
            if time.time() - start_time > timeout:
                print()
                print(f"[KATIB] ❌ Timeout waiting for experiment")
                return False
            
            time.sleep(10)
    
    def export_results(self, experiment_name: str, output_file: str) -> bool:
        """Export experiment results to JSON."""
        try:
            exp = self.get_experiment(experiment_name)
            trials = self.get_trials(experiment_name)
            best = self.get_best_parameters(experiment_name)
            
            results = {
                'experiment': exp,
                'trials': trials,
                'best_parameters': best,
            }
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"[KATIB] Results exported to: {output_file}")
            return True
            
        except Exception as e:
            print(f"[KATIB] Error exporting results: {str(e)}")
            return False
    
    def delete_experiment(self, experiment_name: str) -> bool:
        """Delete experiment from Katib."""
        try:
            self.custom_api.delete_namespaced_custom_object(
                group='kubeflow.org',
                version='v1beta1',
                namespace=self.namespace,
                plural='experiments',
                name=experiment_name,
                body=client.V1DeleteOptions()
            )
            print(f"[KATIB] Experiment deleted: {experiment_name}")
            return True
        except Exception as e:
            print(f"[KATIB] Error deleting experiment: {str(e)}")
            return False
    
    @staticmethod
    def _get_default_trial_template() -> Dict:
        """
        Get default trial template using KFP pipeline.
        
        This template runs the katib_tuning_pipeline.py as a trial worker.
        """
        return {
            'goTemplate': {
                'rawTemplate': '''
apiVersion: batch/v1
kind: Job
metadata:
  name: {{.Trial}}
  namespace: {{.NameSpace}}
spec:
  template:
    spec:
      serviceAccountName: default-editor
      containers:
      - name: training-container
        image: tensorflow/tensorflow:2.13.0-gpu
        imagePullPolicy: IfNotPresent
        command:
        - "python"
        - "-c"
        - |
          import sys
          import os
          
          # Get trial parameters from Katib
          params = {
          {{- range $k, $v := .HyperParameters}}
              "{{$k}}": {{$v}},
          {{- end}}
          }
          
          print(f"[TRIAL] Starting with parameters: {params}")
          
          # Import and run pipeline
          from katib_tuning_pipeline import katib_hyperparameter_tuning_pipeline
          from kfp import dsl, compiler
          
          # Compile and run pipeline with trial parameters
          compiler.Compiler().compile(
              pipeline_func=katib_hyperparameter_tuning_pipeline,
              package_path='trial_pipeline.yaml',
              arguments=params
          )
          
          print("[TRIAL] Pipeline compiled and executed")
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
          requests:
            nvidia.com/gpu: 1
            memory: 4Gi
      restartPolicy: Never
  backoffLimit: 0
'''
            }
        }


def create_learning_rate_tuning_experiment(
    manager: KatibExperimentManager,
    experiment_name: str = 'lr-tuning'
) -> bool:
    """Example: Tune learning rate."""
    print("\n" + "="*70)
    print("KATIB LEARNING RATE TUNING")
    print("="*70)
    
    params = [
        KatibParameter('learning_rate', 'double', {
            'min': '0.0001',
            'max': '0.01'
        }),
        KatibParameter('batch_size', 'int', {
            'min': '32',
            'max': '32'
        }),
    ]
    
    return manager.create_and_submit_experiment(
        name=experiment_name,
        parameters=params,
        trial_count=5,
        parallel_trials=2,
        algorithm='bayesianoptimization'
    )


def create_full_tuning_experiment(
    manager: KatibExperimentManager,
    experiment_name: str = 'full-tuning'
) -> bool:
    """Example: Full hyperparameter tuning."""
    print("\n" + "="*70)
    print("KATIB FULL HYPERPARAMETER TUNING")
    print("="*70)
    
    params = [
        KatibParameter('learning_rate', 'double', {
            'min': '0.0001',
            'max': '0.01'
        }),
        KatibParameter('batch_size', 'categorical', {
            'list': ['16', '32', '64', '128']
        }),
        KatibParameter('dropout_rate', 'double', {
            'min': '0.1',
            'max': '0.5'
        }),
        KatibParameter('num_layers', 'int', {
            'min': '2',
            'max': '5'
        }),
    ]
    
    return manager.create_and_submit_experiment(
        name=experiment_name,
        parameters=params,
        trial_count=20,
        parallel_trials=4,
        algorithm='bayesianoptimization'
    )


def main():
    """Main Katib invocation example."""
    print("\n" + "="*70)
    print("KATIB HYPERPARAMETER TUNING")
    print("="*70 + "\n")
    
    # Initialize manager
    manager = KatibExperimentManager(namespace='kubeflow')
    
    # Create learning rate tuning experiment
    if create_learning_rate_tuning_experiment(manager, 'lr-tuning-exp'):
        print("\n[MAIN] ✅ Experiment submitted successfully!")
        
        # Watch it
        if manager.watch_experiment('lr-tuning-exp', timeout=600):
            # Get results
            best = manager.get_best_parameters('lr-tuning-exp')
            
            # Export results
            manager.export_results('lr-tuning-exp', 'katib-results.json')
    else:
        print("\n[MAIN] ❌ Failed to submit experiment")


if __name__ == '__main__':
    main()

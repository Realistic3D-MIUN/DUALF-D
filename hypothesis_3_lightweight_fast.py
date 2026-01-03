#!/usr/bin/env python3
"""
Hypothesis 3: Lightweight/Fast
===============================

Pipeline: Latent Reordering → Non-Uniform Quantization → Simple Entropy → Flat Bitstream

This hypothesis focuses on computational efficiency and speed with minimal complexity.
Key features:
- Fast channel reordering for basic prioritization
- Efficient Lloyd-Max quantization
- Simple arithmetic entropy coding
- Minimal overhead for real-time applications
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from compression_pipeline import CompletePipelineDecoder
import matplotlib.pyplot as plt


class LightweightFastCoding:
    """
    Implementation of Hypothesis 3: Lightweight/Fast Coding.
    Optimized for speed and computational efficiency with minimal complexity.
    """
    
    def __init__(self, save_dir: str = "results_hypothesis_3"):
        """
        Initialize the Lightweight/Fast Coding pipeline.
        
        Args:
            save_dir: Directory to save results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Pipeline configuration for Lightweight/Fast coding
        self.pipeline_config = {
            'use_reordering': True,
            'use_clipping_sparsification': False,  # Skip for speed
            'use_non_uniform_quantization': True,
            'use_vector_quantization': False,      # Too complex for lightweight
            'use_transform_coding': False,         # Skip DCT for speed
            'use_bit_plane_coding': False,         # Skip for simplicity
            'use_bitstream_structuring': False,    # Skip for speed
            'use_arithmetic_coding': True
        }
        
        # Component configurations optimized for speed
        self.component_configs = {
            'reorderer': {
                'importance_metric': 'variance'  # Fastest metric
            },
            'quantizer': {
                'num_levels': 8,       # Fewer levels for speed
                'channel_wise': False  # Global quantization for speed
            },
            'arithmetic_coder': {
                'channel_wise': False, # Global for speed
                'precision': 16        # Lower precision for speed
            }
        }
        
        self.pipeline = None
        self.training_results = {}
        self.speed_results = {}
    
    def initialize_pipeline(self):
        """Initialize the compression pipeline with lightweight settings."""
        print("🔧 Initializing Lightweight/Fast Coding Pipeline...")
        self.pipeline = CompletePipelineDecoder(pipeline_config=self.pipeline_config)
        self.pipeline.initialize_components(self.component_configs)
        print(f"✅ Pipeline initialized with {sum(self.pipeline_config.values())}/8 active components")
    
    def train_pipeline(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor):
        """
        Train the pipeline with focus on speed.
        
        Args:
            spatial_latents: Training spatial latents [batch, 64, 8, 12]
            angular_latents: Training angular latents [batch, 64, 8, 12]
        """
        if self.pipeline is None:
            self.initialize_pipeline()
        
        print("\n🎓 Training Lightweight/Fast Coding Pipeline...")
        start_time = time.time()
        
        self.pipeline.train_pipeline(spatial_latents, angular_latents)
        
        training_time = time.time() - start_time
        self.training_results = {
            'training_time': training_time,
            'pipeline_config': self.pipeline_config,
            'component_configs': self.component_configs,
            'data_shape': {
                'spatial': list(spatial_latents.shape),
                'angular': list(angular_latents.shape)
            }
        }
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
    
    def benchmark_speed_performance(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor,
                                  num_runs: int = 5):
        """
        Benchmark speed performance with multiple runs.
        
        Args:
            spatial_latents: Input spatial latents
            angular_latents: Input angular latents
            num_runs: Number of benchmark runs
            
        Returns:
            Comprehensive speed benchmarking results
        """
        if self.pipeline is None or not self.pipeline.is_trained:
            raise ValueError("Pipeline must be trained before benchmarking!")
        
        print(f"\n⚡ Speed Benchmarking with {num_runs} runs...")
        
        encoding_times = []
        decoding_times = []
        total_times = []
        
        # Warm-up run
        print("🔥 Warm-up run...")
        try:
            encoded_data, complete_side_info = self.pipeline.encode_complete(spatial_latents, angular_latents)
            reconstructed_spatial, reconstructed_angular = self.pipeline.decode_complete(encoded_data, complete_side_info)
        except Exception as e:
            print(f"❌ Warm-up failed: {str(e)}")
            return {'error': str(e)}
        
        # Benchmark runs
        for run in range(num_runs):
            print(f"📊 Benchmark run {run + 1}/{num_runs}...")
            
            # Encoding benchmark
            start_time = time.time()
            encoded_data, complete_side_info = self.pipeline.encode_complete(spatial_latents, angular_latents)
            encoding_time = time.time() - start_time
            encoding_times.append(encoding_time)
            
            # Decoding benchmark
            start_time = time.time()
            reconstructed_spatial, reconstructed_angular = self.pipeline.decode_complete(encoded_data, complete_side_info)
            decoding_time = time.time() - start_time
            decoding_times.append(decoding_time)
            
            total_time = encoding_time + decoding_time
            total_times.append(total_time)
            
            print(f"  ⏱️ Run {run + 1}: Encoding {encoding_time:.3f}s, Decoding {decoding_time:.3f}s, Total {total_time:.3f}s")
        
        # Calculate statistics
        encoding_stats = self._calculate_timing_stats(encoding_times)
        decoding_stats = self._calculate_timing_stats(decoding_times)
        total_stats = self._calculate_timing_stats(total_times)
        
        # Performance analysis
        performance = self.pipeline.calculate_end_to_end_performance(
            spatial_latents, angular_latents, encoded_data, complete_side_info
        )
        
        # Calculate throughput
        data_elements = spatial_latents.numel() + angular_latents.numel()
        encoding_throughput = data_elements / encoding_stats['mean']  # elements/second
        decoding_throughput = data_elements / decoding_stats['mean']
        
        results = {
            'benchmark_settings': {
                'num_runs': num_runs,
                'data_elements': data_elements,
                'data_size_mb': (data_elements * 4) / (1024 * 1024)  # float32 = 4 bytes
            },
            'timing_statistics': {
                'encoding': encoding_stats,
                'decoding': decoding_stats,
                'total': total_stats
            },
            'throughput': {
                'encoding_elements_per_sec': encoding_throughput,
                'decoding_elements_per_sec': decoding_throughput,
                'encoding_mb_per_sec': (encoding_throughput * 4) / (1024 * 1024),
                'decoding_mb_per_sec': (decoding_throughput * 4) / (1024 * 1024)
            },
            'compression_performance': performance,
            'efficiency_metrics': {
                'speed_efficiency': self._assess_speed_efficiency(total_stats['mean']),
                'compression_efficiency': self._assess_compression_efficiency(performance['compression_ratio']),
                'memory_efficiency': self._assess_memory_efficiency(),
                'overall_rating': self._calculate_overall_rating(total_stats['mean'], performance['compression_ratio'])
            }
        }
        
        self.speed_results = results
        return results
    
    def _calculate_timing_stats(self, times):
        """Calculate timing statistics."""
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'cv': np.std(times) / np.mean(times) if np.mean(times) > 0 else 0  # Coefficient of variation
        }
    
    def _assess_speed_efficiency(self, total_time):
        """Assess speed efficiency."""
        if total_time < 1.0:
            return 'excellent'
        elif total_time < 3.0:
            return 'good'
        elif total_time < 5.0:
            return 'moderate'
        else:
            return 'slow'
    
    def _assess_compression_efficiency(self, compression_ratio):
        """Assess compression efficiency."""
        if compression_ratio > 15:
            return 'excellent'
        elif compression_ratio > 10:
            return 'good'
        elif compression_ratio > 5:
            return 'moderate'
        else:
            return 'poor'
    
    def _assess_memory_efficiency(self):
        """Assess memory efficiency (simplified)."""
        # Lightweight pipeline has minimal memory overhead
        active_components = sum(self.pipeline_config.values())
        if active_components <= 3:
            return 'excellent'
        elif active_components <= 5:
            return 'good'
        else:
            return 'moderate'
    
    def _calculate_overall_rating(self, total_time, compression_ratio):
        """Calculate overall performance rating."""
        speed_score = 4 if total_time < 1.0 else 3 if total_time < 3.0 else 2 if total_time < 5.0 else 1
        compression_score = 4 if compression_ratio > 15 else 3 if compression_ratio > 10 else 2 if compression_ratio > 5 else 1
        
        overall_score = (speed_score + compression_score) / 2
        
        if overall_score >= 3.5:
            return 'excellent'
        elif overall_score >= 2.5:
            return 'good'
        elif overall_score >= 1.5:
            return 'moderate'
        else:
            return 'poor'
    
    def analyze_component_efficiency(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor):
        """
        Analyze efficiency of individual components.
        
        Args:
            spatial_latents: Input spatial latents
            angular_latents: Input angular latents
            
        Returns:
            Component efficiency analysis
        """
        if self.pipeline is None or not self.pipeline.is_trained:
            raise ValueError("Pipeline must be trained first!")
        
        print("\n🔍 Analyzing Component Efficiency...")
        
        analysis = {}
        
        # Reordering efficiency
        if self.pipeline.reorderer:
            print("📊 Analyzing Latent Reordering...")
            reorder_start = time.time()
            reordered_spatial, reordered_angular, reorder_info = self.pipeline.reorderer.reorder(
                spatial_latents, angular_latents
            )
            reorder_time = time.time() - reorder_start
            
            analysis['reordering'] = {
                'processing_time': reorder_time,
                'importance_metric': self.pipeline.reorderer.importance_metric,
                'spatial_importance_range': [
                    float(np.min(self.pipeline.reorderer.spatial_importance)),
                    float(np.max(self.pipeline.reorderer.spatial_importance))
                ],
                'angular_importance_range': [
                    float(np.min(self.pipeline.reorderer.angular_importance)),
                    float(np.max(self.pipeline.reorderer.angular_importance))
                ],
                'efficiency': 'excellent' if reorder_time < 0.1 else 'good' if reorder_time < 0.5 else 'moderate'
            }
        
        # Quantization efficiency
        if self.pipeline.quantizer:
            print("📊 Analyzing Non-Uniform Quantization...")
            quant_start = time.time()
            quantized_spatial, quantized_angular, quant_info = self.pipeline.quantizer.quantize(
                spatial_latents, angular_latents
            )
            quant_time = time.time() - quant_start
            
            analysis['quantization'] = {
                'processing_time': quant_time,
                'num_levels': self.component_configs['quantizer']['num_levels'],
                'channel_wise': self.component_configs['quantizer']['channel_wise'],
                'compression_stats': quant_info.get('compression_stats', {}),
                'efficiency': 'excellent' if quant_time < 0.5 else 'good' if quant_time < 1.0 else 'moderate'
            }
        
        # Arithmetic coding efficiency
        if self.pipeline.arithmetic_coder:
            print("📊 Analyzing Arithmetic Entropy Coding...")
            # Use quantized data if available, otherwise original
            test_spatial = quantized_spatial if 'quantized_spatial' in locals() else spatial_latents
            test_angular = quantized_angular if 'quantized_angular' in locals() else angular_latents
            
            arithmetic_start = time.time()
            compressed_bitstream, arithmetic_info = self.pipeline.arithmetic_coder.encode(
                test_spatial, test_angular
            )
            arithmetic_time = time.time() - arithmetic_start
            
            analysis['arithmetic_coding'] = {
                'processing_time': arithmetic_time,
                'channel_wise': self.component_configs['arithmetic_coder']['channel_wise'],
                'precision': self.component_configs['arithmetic_coder']['precision'],
                'compression_stats': arithmetic_info.get('compression_stats', {}),
                'efficiency': 'excellent' if arithmetic_time < 1.0 else 'good' if arithmetic_time < 2.0 else 'moderate'
            }
        
        return analysis
    
    def generate_speed_report(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor, num_runs: int = 5):
        """
        Generate comprehensive speed analysis report.
        
        Args:
            spatial_latents: Input spatial latents
            angular_latents: Input angular latents
            num_runs: Number of benchmark runs
        """
        print("\n📋 Generating Lightweight/Fast Speed Report...")
        
        # Perform speed benchmarking
        if not hasattr(self, 'speed_results') or not self.speed_results:
            self.benchmark_speed_performance(spatial_latents, angular_latents, num_runs)
        
        # Component efficiency analysis
        component_analysis = self.analyze_component_efficiency(spatial_latents, angular_latents)
        
        # Create comprehensive report
        report = {
            'hypothesis': 'Lightweight/Fast Coding',
            'description': 'Optimized for speed and computational efficiency with minimal complexity',
            'pipeline_components': [
                'Latent Reordering (variance-based)',
                'Non-Uniform Quantization (8 levels, global)',
                'Arithmetic Entropy Coding (16-bit precision)'
            ],
            'training_results': self.training_results,
            'speed_results': self.speed_results,
            'component_analysis': component_analysis,
            'optimization_insights': self._generate_optimization_insights(),
            'performance_comparison': self._generate_performance_comparison(),
            'recommendations': self._generate_speed_recommendations()
        }
        
        # Save report
        report_path = self.save_dir / 'speed_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to {report_path}")
        
        # Generate speed visualizations
        self._generate_speed_visualizations(component_analysis)
        
        return report
    
    def _generate_optimization_insights(self):
        """Generate optimization insights based on results."""
        insights = []
        
        if hasattr(self, 'speed_results') and 'timing_statistics' in self.speed_results:
            timing = self.speed_results['timing_statistics']
            
            # Encoding vs decoding analysis
            enc_mean = timing['encoding']['mean']
            dec_mean = timing['decoding']['mean']
            
            if enc_mean > dec_mean * 1.5:
                insights.append("🔍 Encoding is bottleneck - consider optimizing quantization training")
            elif dec_mean > enc_mean * 1.5:
                insights.append("🔍 Decoding is bottleneck - consider simplifying arithmetic decoding")
            else:
                insights.append("✅ Well-balanced encoding/decoding performance")
            
            # Consistency analysis
            total_cv = timing['total']['cv']
            if total_cv < 0.05:
                insights.append("📊 Excellent timing consistency - suitable for real-time applications")
            elif total_cv < 0.1:
                insights.append("📊 Good timing consistency - suitable for most applications")
            else:
                insights.append("⚠️ Variable timing - may need optimization for real-time use")
        
        # Component analysis insights
        if hasattr(self, 'speed_results') and 'efficiency_metrics' in self.speed_results:
            metrics = self.speed_results['efficiency_metrics']
            if metrics['speed_efficiency'] == 'excellent' and metrics['compression_efficiency'] == 'excellent':
                insights.append("🏆 Optimal balance of speed and compression achieved")
            elif metrics['speed_efficiency'] == 'excellent':
                insights.append("⚡ Excellent speed - ideal for real-time applications")
            elif metrics['compression_efficiency'] == 'excellent':
                insights.append("🗜️ Excellent compression despite lightweight design")
        
        return insights
    
    def _generate_performance_comparison(self):
        """Generate performance comparison with theoretical limits."""
        if not hasattr(self, 'speed_results'):
            return {}
        
        speed_results = self.speed_results
        
        # Theoretical estimates for comparison
        data_elements = speed_results['benchmark_settings']['data_elements']
        
        # Estimate theoretical minimum time (based on memory bandwidth)
        # Assuming ~10 GB/s memory bandwidth, and 4 bytes per float
        theoretical_min_time = (data_elements * 4) / (10 * 1024 * 1024 * 1024)
        
        actual_time = speed_results['timing_statistics']['total']['mean']
        efficiency_ratio = theoretical_min_time / actual_time
        
        return {
            'theoretical_minimum_time': theoretical_min_time,
            'actual_time': actual_time,
            'efficiency_ratio': efficiency_ratio,
            'overhead_factor': actual_time / theoretical_min_time,
            'performance_class': 'optimal' if efficiency_ratio > 0.1 else 'good' if efficiency_ratio > 0.05 else 'moderate'
        }
    
    def _generate_speed_recommendations(self):
        """Generate speed-focused recommendations."""
        recommendations = [
            "⚡ Minimal components: Only essential compression stages",
            "🎯 Global quantization: Faster than channel-wise processing",
            "💾 Low precision: 16-bit arithmetic for speed",
            "🔄 Simple pipeline: Easy to optimize and parallelize"
        ]
        
        if hasattr(self, 'speed_results'):
            speed_efficiency = self.speed_results.get('efficiency_metrics', {}).get('speed_efficiency', '')
            if speed_efficiency == 'excellent':
                recommendations.append("🏆 Excellent speed achieved - suitable for real-time streaming")
            elif speed_efficiency == 'good':
                recommendations.append("✅ Good speed - suitable for interactive applications")
            
            throughput = self.speed_results.get('throughput', {})
            if throughput.get('encoding_mb_per_sec', 0) > 100:
                recommendations.append("🚀 High throughput achieved - suitable for high-resolution data")
        
        return recommendations
    
    def _generate_speed_visualizations(self, component_analysis):
        """Generate speed-focused visualizations."""
        try:
            plt.figure(figsize=(15, 10))
            
            # Timing breakdown
            if hasattr(self, 'speed_results') and 'timing_statistics' in self.speed_results:
                plt.subplot(2, 2, 1)
                timing = self.speed_results['timing_statistics']
                
                categories = ['Encoding', 'Decoding']
                means = [timing['encoding']['mean'], timing['decoding']['mean']]
                stds = [timing['encoding']['std'], timing['decoding']['std']]
                
                bars = plt.bar(categories, means, yerr=stds, capsize=5, 
                              color=['lightblue', 'lightcoral'], alpha=0.7)
                plt.ylabel('Time (seconds)')
                plt.title('Processing Time Breakdown')
                
                # Add value labels
                for bar, mean in zip(bars, means):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.1,
                            f'{mean:.3f}s', ha='center', va='bottom')
            
            # Component timing
            if component_analysis:
                plt.subplot(2, 2, 2)
                components = []
                times = []
                
                for comp_name, comp_data in component_analysis.items():
                    if 'processing_time' in comp_data:
                        components.append(comp_name.replace('_', '\n').title())
                        times.append(comp_data['processing_time'])
                
                if components and times:
                    bars = plt.bar(components, times, color=['green', 'orange', 'purple'][:len(components)])
                    plt.ylabel('Time (seconds)')
                    plt.title('Component Processing Times')
                    plt.xticks(rotation=45)
                    
                    # Add value labels
                    for bar, time_val in zip(bars, times):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                                f'{time_val:.3f}s', ha='center', va='bottom')
            
            # Throughput analysis
            if hasattr(self, 'speed_results') and 'throughput' in self.speed_results:
                plt.subplot(2, 2, 3)
                throughput = self.speed_results['throughput']
                
                operations = ['Encoding', 'Decoding']
                mb_per_sec = [
                    throughput.get('encoding_mb_per_sec', 0),
                    throughput.get('decoding_mb_per_sec', 0)
                ]
                
                bars = plt.bar(operations, mb_per_sec, color=['skyblue', 'salmon'])
                plt.ylabel('Throughput (MB/s)')
                plt.title('Data Throughput')
                
                # Add value labels
                for bar, throughput_val in zip(bars, mb_per_sec):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mb_per_sec)*0.01,
                            f'{throughput_val:.1f}', ha='center', va='bottom')
            
            # Performance consistency
            if hasattr(self, 'speed_results') and 'timing_statistics' in self.speed_results:
                plt.subplot(2, 2, 4)
                timing = self.speed_results['timing_statistics']
                
                metrics = ['Mean', 'Std Dev', 'CV %']
                values = [
                    timing['total']['mean'],
                    timing['total']['std'],
                    timing['total']['cv'] * 100
                ]
                
                bars = plt.bar(metrics, values, color=['gold', 'lightgreen', 'lightcoral'])
                plt.ylabel('Value')
                plt.title('Performance Consistency')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                            f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'speed_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Speed visualizations saved to {self.save_dir}")
            
        except Exception as e:
            print(f"⚠️ Could not generate speed visualizations: {str(e)}")


def test_lightweight_fast_coding():
    """Test the Lightweight/Fast Coding hypothesis."""
    print("=" * 80)
    print("🎯 TESTING HYPOTHESIS 3: LIGHTWEIGHT/FAST CODING")
    print("=" * 80)
    
    # Generate test data
    print("📊 Generating test data...")
    torch.manual_seed(42)
    batch_size, channels, height, width = 1, 64, 8, 12
    
    spatial_latents = torch.randn(batch_size, channels, height, width)
    angular_latents = torch.randn(batch_size, channels, height, width)
    
    # Keep data relatively simple for speed testing
    spatial_latents *= 0.5  # Reduce dynamic range
    angular_latents *= 0.5
    
    print(f"✅ Generated latents: {spatial_latents.shape}")
    
    # Initialize and test the hypothesis
    lightweight_coder = LightweightFastCoding(save_dir="results_hypothesis_3")
    
    # Train the pipeline
    lightweight_coder.train_pipeline(spatial_latents, angular_latents)
    
    # Benchmark speed performance
    speed_results = lightweight_coder.benchmark_speed_performance(spatial_latents, angular_latents, num_runs=5)
    
    # Generate comprehensive report
    report = lightweight_coder.generate_speed_report(spatial_latents, angular_latents, num_runs=5)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 LIGHTWEIGHT/FAST CODING RESULTS SUMMARY")
    print("=" * 80)
    
    if 'error' not in speed_results:
        timing = speed_results.get('timing_statistics', {})
        throughput = speed_results.get('throughput', {})
        efficiency = speed_results.get('efficiency_metrics', {})
        compression = speed_results.get('compression_performance', {})
        
        print(f"⚡ Total Time: {timing.get('total', {}).get('mean', 0):.3f} ± {timing.get('total', {}).get('std', 0):.3f}s")
        print(f"📤 Encoding Time: {timing.get('encoding', {}).get('mean', 0):.3f}s")
        print(f"📥 Decoding Time: {timing.get('decoding', {}).get('mean', 0):.3f}s")
        print(f"🚀 Encoding Throughput: {throughput.get('encoding_mb_per_sec', 0):.1f} MB/s")
        print(f"🚀 Decoding Throughput: {throughput.get('decoding_mb_per_sec', 0):.1f} MB/s")
        print(f"🎯 Compression Ratio: {compression.get('compression_ratio', 0):.2f}x")
        print(f"📉 Size Reduction: {compression.get('size_reduction_percent', 0):.1f}%")
        print(f"⚡ Speed Efficiency: {efficiency.get('speed_efficiency', 'unknown')}")
        print(f"🗜️ Compression Efficiency: {efficiency.get('compression_efficiency', 'unknown')}")
        print(f"🏆 Overall Rating: {efficiency.get('overall_rating', 'unknown')}")
        
        # Performance consistency
        cv = timing.get('total', {}).get('cv', 0)
        print(f"📊 Timing Consistency: {cv*100:.1f}% CV ({'Excellent' if cv < 0.05 else 'Good' if cv < 0.1 else 'Variable'})")
        
    else:
        print(f"❌ Speed benchmark failed: {speed_results['error']}")
    
    print(f"\n🎯 Pipeline Components: {len(lightweight_coder.pipeline_config)} configured")
    print(f"📁 Results saved to: {lightweight_coder.save_dir}")
    print(f"⏱️  Training time: {lightweight_coder.training_results['training_time']:.2f}s")
    
    print("\n✅ Lightweight/Fast Coding hypothesis test completed!")
    return lightweight_coder, report


if __name__ == "__main__":
    test_lightweight_fast_coding() 
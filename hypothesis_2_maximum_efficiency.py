#!/usr/bin/env python3
"""
Hypothesis 2: Maximum Efficiency
================================

Pipeline: Clipping/Sparsification → Transform Coding (4×4 DCT) → Vector Quantization → Entropy Modeling/Arithmetic → RLE+Zigzag → Bitstream Structuring

This hypothesis focuses on achieving maximum compression efficiency through advanced signal processing techniques.
Key features:
- Aggressive outlier clipping and sparsity promotion
- DCT transform coding for energy compaction
- Vector quantization for spatial correlation exploitation
- Advanced entropy modeling
- Comprehensive bitstream structuring (RLE, zigzag, Rice coding)
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from compression_pipeline import CompletePipelineDecoder
import matplotlib.pyplot as plt


class MaximumEfficiencyCoding:
    """
    Implementation of Hypothesis 2: Maximum Efficiency Coding.
    Optimized for achieving the highest possible compression ratios.
    """
    
    def __init__(self, save_dir: str = "results_hypothesis_2"):
        """
        Initialize the Maximum Efficiency Coding pipeline.
        
        Args:
            save_dir: Directory to save results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Pipeline configuration for Maximum Efficiency
        self.pipeline_config = {
            'use_reordering': False,  # Focus on transform efficiency
            'use_clipping_sparsification': True,
            'use_non_uniform_quantization': False,
            'use_vector_quantization': True,
            'use_transform_coding': True,
            'use_bit_plane_coding': False,
            'use_bitstream_structuring': True,
            'use_arithmetic_coding': False  # Use bitstream structuring instead
        }
        
        # Component configurations optimized for maximum efficiency
        self.component_configs = {
            'clipper_sparsifier': {
                'clipping_method': 'adaptive',  # Most aggressive clipping
                'sparsity_method': 'adaptive',   # Adaptive sparsity promotion
                'clip_percentile': 98.0,        # Aggressive outlier removal
                'sparsity_threshold': 0.2       # High sparsity threshold
            },
            'transform_coder': {
                'block_size': 4,
                'channel_wise': True,
                'use_adaptive_quantization': True,
                'quantization_matrix': None  # Will use JPEG-like matrix
            },
            'vector_quantizer': {
                'codebook_size': 256,  # Large codebook for better quality
                'vector_dim': 4,       # 2×2 patches
                'overlap_stride': 2,
                'channel_wise': True,
                'max_iterations': 150  # More iterations for better convergence
            },
            'bitstream_structurer': {
                'use_rle': True,
                'use_zigzag': True,
                'use_rice_coding': True,
                'rice_parameter': 1,   # Optimized for sparse data
                'channel_wise': True
            }
        }
        
        self.pipeline = None
        self.training_results = {}
        self.efficiency_results = {}
    
    def initialize_pipeline(self):
        """Initialize the compression pipeline with maximum efficiency settings."""
        print("🔧 Initializing Maximum Efficiency Coding Pipeline...")
        self.pipeline = CompletePipelineDecoder(pipeline_config=self.pipeline_config)
        self.pipeline.initialize_components(self.component_configs)
        print(f"✅ Pipeline initialized with {sum(self.pipeline_config.values())}/8 active components")
    
    def train_pipeline(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor):
        """
        Train the pipeline for maximum efficiency.
        
        Args:
            spatial_latents: Training spatial latents [batch, 64, 8, 12]
            angular_latents: Training angular latents [batch, 64, 8, 12]
        """
        if self.pipeline is None:
            self.initialize_pipeline()
        
        print("\n🎓 Training Maximum Efficiency Coding Pipeline...")
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
    
    def analyze_efficiency_components(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor):
        """
        Analyze the efficiency contribution of each component.
        
        Args:
            spatial_latents: Input spatial latents
            angular_latents: Input angular latents
            
        Returns:
            Dictionary with detailed efficiency analysis
        """
        if self.pipeline is None or not self.pipeline.is_trained:
            raise ValueError("Pipeline must be trained first!")
        
        print("\n🔍 Analyzing Component Efficiency Contributions...")
        
        analysis = {}
        current_spatial = spatial_latents.clone()
        current_angular = angular_latents.clone()
        
        original_size = current_spatial.numel() + current_angular.numel()
        
        # Step 1: Clipping and Sparsification Analysis
        if self.pipeline.clipper_sparsifier:
            print("📊 Analyzing Clipping & Sparsification...")
            processed_spatial, processed_angular, clip_info = self.pipeline.clipper_sparsifier.apply_clipping_sparsification(
                current_spatial, current_angular
            )
            
            sparsity_benefit = self.pipeline.clipper_sparsifier.calculate_compression_benefit(
                current_spatial, current_angular, processed_spatial, processed_angular
            )
            
            analysis['clipping_sparsification'] = {
                'spatial_sparsity_ratio': clip_info['spatial_sparsity_stats']['sparsity_ratio'],
                'angular_sparsity_ratio': clip_info['angular_sparsity_stats']['sparsity_ratio'],
                'spatial_outlier_ratio': clip_info['spatial_clipping_stats']['outlier_ratio'],
                'angular_outlier_ratio': clip_info['angular_clipping_stats']['outlier_ratio'],
                'estimated_compression_benefit': sparsity_benefit
            }
            
            current_spatial, current_angular = processed_spatial, processed_angular
        
        # Step 2: Transform Coding Analysis
        if self.pipeline.transform_coder:
            print("📊 Analyzing Transform Coding...")
            dct_spatial, dct_angular, transform_info = self.pipeline.transform_coder.apply_transform(
                current_spatial, current_angular
            )
            
            transform_benefit = self.pipeline.transform_coder.calculate_compression_benefit(
                current_spatial, current_angular, dct_spatial, dct_angular, transform_info
            )
            
            analysis['transform_coding'] = {
                'energy_compaction_ratio': transform_info['transform_stats']['energy_compaction_ratio'],
                'spatial_sparsity_ratio': transform_info['transform_stats']['spatial_sparsity_ratio'],
                'angular_sparsity_ratio': transform_info['transform_stats']['angular_sparsity_ratio'],
                'compression_benefit': transform_benefit
            }
            
            current_spatial, current_angular = dct_spatial, dct_angular
        
        # Step 3: Vector Quantization Analysis
        if self.pipeline.vector_quantizer:
            print("📊 Analyzing Vector Quantization...")
            spatial_indices, angular_indices, vq_info = self.pipeline.vector_quantizer.quantize(
                current_spatial, current_angular
            )
            
            vq_compression = self.pipeline.vector_quantizer.calculate_compression_ratio(
                current_spatial, current_angular, spatial_indices, angular_indices, vq_info
            )
            
            analysis['vector_quantization'] = {
                'codebook_size': vq_info['codebook_size'],
                'vector_dimension': vq_info['vector_dim'],
                'spatial_patches': vq_info['spatial_total_patches'],
                'angular_patches': vq_info['angular_total_patches'],
                'compression_stats': vq_compression
            }
        
        # Step 4: Bitstream Structuring Analysis
        if self.pipeline.bitstream_structurer:
            print("📊 Analyzing Bitstream Structuring...")
            structured_data, struct_info = self.pipeline.bitstream_structurer.structure_bitstream(
                current_spatial, current_angular
            )
            
            struct_benefit = self.pipeline.bitstream_structurer.calculate_compression_benefit(
                current_spatial, current_angular, structured_data, struct_info
            )
            
            analysis['bitstream_structuring'] = {
                'rle_efficiency': struct_info['compression_stats']['compression_ratio'],
                'spatial_zero_ratio': struct_info['spatial_patterns']['zero_ratio'],
                'angular_zero_ratio': struct_info['angular_patterns']['zero_ratio'],
                'rice_parameter': struct_info['rice_parameter'],
                'compression_benefit': struct_benefit
            }
        
        return analysis
    
    def compress_with_efficiency_focus(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor):
        """
        Perform compression with focus on maximum efficiency.
        
        Args:
            spatial_latents: Input spatial latents
            angular_latents: Input angular latents
            
        Returns:
            Comprehensive efficiency results
        """
        if self.pipeline is None or not self.pipeline.is_trained:
            raise ValueError("Pipeline must be trained before compression!")
        
        print("\n📦 Maximum Efficiency Compression...")
        
        # Full compression analysis
        start_time = time.time()
        
        try:
            encoded_data, complete_side_info = self.pipeline.encode_complete(spatial_latents, angular_latents)
            encoding_time = time.time() - start_time
            
            # Detailed decoding analysis
            start_time = time.time()
            reconstructed_spatial, reconstructed_angular = self.pipeline.decode_complete(
                encoded_data, complete_side_info
            )
            decoding_time = time.time() - start_time
            
            # Performance metrics
            performance = self.pipeline.calculate_end_to_end_performance(
                spatial_latents, angular_latents, encoded_data, complete_side_info
            )
            
            # Component efficiency analysis
            component_analysis = self.analyze_efficiency_components(spatial_latents, angular_latents)
            
            results = {
                'overall_performance': performance,
                'timing': {
                    'encoding_time': encoding_time,
                    'decoding_time': decoding_time,
                    'total_time': encoding_time + decoding_time
                },
                'component_analysis': component_analysis,
                'efficiency_metrics': {
                    'compression_ratio': performance['compression_ratio'],
                    'size_reduction': performance['size_reduction_percent'],
                    'bits_per_element': performance['bits_per_element'],
                    'reconstruction_fidelity': self._calculate_fidelity(
                        spatial_latents, angular_latents,
                        reconstructed_spatial, reconstructed_angular
                    )
                }
            }
            
            self.efficiency_results = results
            return results
            
        except Exception as e:
            print(f"❌ Compression failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_fidelity(self, orig_spatial, orig_angular, recon_spatial, recon_angular):
        """Calculate detailed reconstruction fidelity metrics."""
        spatial_mse = torch.mean((orig_spatial - recon_spatial) ** 2).item()
        angular_mse = torch.mean((orig_angular - recon_angular) ** 2).item()
        
        # Signal-to-noise ratio
        spatial_signal_power = torch.mean(orig_spatial ** 2).item()
        angular_signal_power = torch.mean(orig_angular ** 2).item()
        
        spatial_snr = 10 * np.log10(spatial_signal_power / max(spatial_mse, 1e-10))
        angular_snr = 10 * np.log10(angular_signal_power / max(angular_mse, 1e-10))
        
        # Structural similarity (simplified)
        spatial_corr = torch.corrcoef(torch.stack([
            orig_spatial.flatten(), recon_spatial.flatten()
        ]))[0, 1].item()
        angular_corr = torch.corrcoef(torch.stack([
            orig_angular.flatten(), recon_angular.flatten()
        ]))[0, 1].item()
        
        return {
            'spatial_mse': spatial_mse,
            'angular_mse': angular_mse,
            'spatial_snr_db': spatial_snr,
            'angular_snr_db': angular_snr,
            'spatial_correlation': spatial_corr,
            'angular_correlation': angular_corr,
            'overall_fidelity': 'excellent' if spatial_snr > 30 and angular_snr > 30 else 'good' if spatial_snr > 20 and angular_snr > 20 else 'moderate'
        }
    
    def generate_efficiency_report(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor):
        """
        Generate comprehensive efficiency analysis report.
        
        Args:
            spatial_latents: Input spatial latents
            angular_latents: Input angular latents
        """
        print("\n📋 Generating Maximum Efficiency Report...")
        
        # Perform efficiency analysis
        if not hasattr(self, 'efficiency_results') or not self.efficiency_results:
            self.compress_with_efficiency_focus(spatial_latents, angular_latents)
        
        # Create comprehensive report
        report = {
            'hypothesis': 'Maximum Efficiency Coding',
            'description': 'Optimized for achieving highest possible compression ratios',
            'pipeline_components': [
                'Clipping & Sparsification (adaptive)',
                'Transform Coding (4×4 DCT)',
                'Vector Quantization (256 codebook)',
                'Bitstream Structuring (RLE + Zigzag + Rice)'
            ],
            'training_results': self.training_results,
            'efficiency_results': self.efficiency_results,
            'optimization_strategies': self._generate_optimization_strategies(),
            'performance_analysis': self._generate_performance_analysis(),
            'recommendations': self._generate_efficiency_recommendations()
        }
        
        # Save report
        report_path = self.save_dir / 'efficiency_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to {report_path}")
        
        # Generate efficiency visualizations
        self._generate_efficiency_visualizations()
        
        return report
    
    def _generate_optimization_strategies(self):
        """Generate optimization strategies based on component analysis."""
        strategies = []
        
        if hasattr(self, 'efficiency_results') and 'component_analysis' in self.efficiency_results:
            comp_analysis = self.efficiency_results['component_analysis']
            
            # Clipping/Sparsification optimization
            if 'clipping_sparsification' in comp_analysis:
                sparsity = comp_analysis['clipping_sparsification']
                if sparsity.get('spatial_sparsity_ratio', 0) > 0.5:
                    strategies.append("🎯 High sparsity detected - RLE will be very effective")
                if sparsity.get('spatial_outlier_ratio', 0) > 0.1:
                    strategies.append("✂️ Significant outliers present - aggressive clipping beneficial")
            
            # Transform coding optimization
            if 'transform_coding' in comp_analysis:
                transform = comp_analysis['transform_coding']
                if transform.get('energy_compaction_ratio', 0) > 0.3:
                    strategies.append("⚡ Good energy compaction - DCT transform very effective")
            
            # Vector quantization optimization
            if 'vector_quantization' in comp_analysis:
                vq = comp_analysis['vector_quantization']
                compression_stats = vq.get('compression_stats', {})
                if compression_stats.get('compression_ratio', 0) > 15:
                    strategies.append("🎨 Vector quantization achieving excellent compression")
        
        return strategies
    
    def _generate_performance_analysis(self):
        """Generate detailed performance analysis."""
        if not hasattr(self, 'efficiency_results') or 'overall_performance' not in self.efficiency_results:
            return {}
        
        perf = self.efficiency_results['overall_performance']
        
        return {
            'compression_efficiency': 'excellent' if perf['compression_ratio'] > 15 else 'good' if perf['compression_ratio'] > 10 else 'moderate',
            'reconstruction_quality': perf['reconstruction_quality'],
            'speed_efficiency': 'fast' if self.efficiency_results['timing']['total_time'] < 10 else 'moderate',
            'component_contributions': self._analyze_component_contributions(),
            'bottleneck_analysis': self._identify_bottlenecks()
        }
    
    def _analyze_component_contributions(self):
        """Analyze individual component contributions to efficiency."""
        if not hasattr(self, 'efficiency_results') or 'component_analysis' not in self.efficiency_results:
            return {}
        
        comp_analysis = self.efficiency_results['component_analysis']
        contributions = {}
        
        # Estimate contribution of each component
        if 'clipping_sparsification' in comp_analysis:
            sparsity_ratio = comp_analysis['clipping_sparsification'].get('spatial_sparsity_ratio', 0)
            contributions['sparsification'] = f"~{sparsity_ratio*100:.1f}% sparsity contribution"
        
        if 'transform_coding' in comp_analysis:
            energy_ratio = comp_analysis['transform_coding'].get('energy_compaction_ratio', 0)
            contributions['transform'] = f"~{energy_ratio*100:.1f}% energy compaction"
        
        if 'vector_quantization' in comp_analysis:
            vq_stats = comp_analysis['vector_quantization'].get('compression_stats', {})
            vq_ratio = vq_stats.get('compression_ratio', 1)
            contributions['vector_quantization'] = f"~{vq_ratio:.1f}x compression"
        
        return contributions
    
    def _identify_bottlenecks(self):
        """Identify potential bottlenecks in the pipeline."""
        bottlenecks = []
        
        if hasattr(self, 'efficiency_results'):
            timing = self.efficiency_results.get('timing', {})
            if timing.get('encoding_time', 0) > 5:
                bottlenecks.append("Encoding time could be optimized")
            if timing.get('decoding_time', 0) > 5:
                bottlenecks.append("Decoding time could be optimized")
        
        return bottlenecks if bottlenecks else ["No significant bottlenecks detected"]
    
    def _generate_efficiency_recommendations(self):
        """Generate recommendations for maximum efficiency."""
        recommendations = [
            "🎯 Transform coding: Excellent for natural signal compression",
            "🎨 Vector quantization: Exploits spatial correlations effectively",
            "🗜️ Bitstream structuring: Essential for sparse data",
            "⚡ Pipeline order: Optimized for maximum compression gain"
        ]
        
        if hasattr(self, 'efficiency_results') and 'overall_performance' in self.efficiency_results:
            ratio = self.efficiency_results['overall_performance'].get('compression_ratio', 0)
            if ratio > 20:
                recommendations.append("🏆 Exceptional efficiency achieved - suitable for storage applications")
            elif ratio > 15:
                recommendations.append("✅ Excellent efficiency - good for bandwidth-limited transmission")
        
        return recommendations
    
    def _generate_efficiency_visualizations(self):
        """Generate efficiency-focused visualizations."""
        try:
            if not hasattr(self, 'efficiency_results') or not self.efficiency_results:
                return
            
            plt.figure(figsize=(15, 10))
            
            # Component efficiency breakdown
            if 'component_analysis' in self.efficiency_results:
                comp_analysis = self.efficiency_results['component_analysis']
                
                # Plot component contributions
                plt.subplot(2, 2, 1)
                components = []
                efficiencies = []
                
                if 'clipping_sparsification' in comp_analysis:
                    components.append('Clipping\n& Sparsification')
                    sparsity = comp_analysis['clipping_sparsification'].get('spatial_sparsity_ratio', 0)
                    efficiencies.append(sparsity * 100)
                
                if 'transform_coding' in comp_analysis:
                    components.append('Transform\nCoding')
                    energy_compact = comp_analysis['transform_coding'].get('energy_compaction_ratio', 0)
                    efficiencies.append(energy_compact * 100)
                
                if 'vector_quantization' in comp_analysis:
                    components.append('Vector\nQuantization')
                    vq_stats = comp_analysis['vector_quantization'].get('compression_stats', {})
                    vq_eff = min(vq_stats.get('size_reduction_percent', 0), 100)
                    efficiencies.append(vq_eff)
                
                if 'bitstream_structuring' in comp_analysis:
                    components.append('Bitstream\nStructuring')
                    struct_benefit = comp_analysis['bitstream_structuring'].get('compression_benefit', {})
                    struct_eff = min(struct_benefit.get('size_reduction_percent', 0), 100)
                    efficiencies.append(struct_eff)
                
                if components and efficiencies:
                    bars = plt.bar(components, efficiencies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
                    plt.ylabel('Efficiency (%)')
                    plt.title('Component Efficiency Contributions')
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, efficiencies):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{value:.1f}%', ha='center', va='bottom')
            
            # Timing analysis
            if 'timing' in self.efficiency_results:
                plt.subplot(2, 2, 2)
                timing = self.efficiency_results['timing']
                times = [timing.get('encoding_time', 0), timing.get('decoding_time', 0)]
                labels = ['Encoding', 'Decoding']
                colors = ['lightblue', 'lightcoral']
                
                wedges, texts, autotexts = plt.pie(times, labels=labels, colors=colors, autopct='%1.1f%%')
                plt.title('Processing Time Distribution')
            
            # Performance metrics
            if 'overall_performance' in self.efficiency_results:
                plt.subplot(2, 2, 3)
                perf = self.efficiency_results['overall_performance']
                
                metrics = ['Compression\nRatio', 'Size\nReduction %', 'Bits per\nElement']
                values = [
                    perf.get('compression_ratio', 0),
                    perf.get('size_reduction_percent', 0),
                    perf.get('bits_per_element', 32)  # Original is 32 bits
                ]
                
                bars = plt.bar(metrics, values, color=['green', 'blue', 'orange'])
                plt.ylabel('Value')
                plt.title('Overall Performance Metrics')
                plt.xticks(rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                            f'{value:.1f}', ha='center', va='bottom')
            
            # Fidelity metrics
            if 'efficiency_metrics' in self.efficiency_results and 'reconstruction_fidelity' in self.efficiency_results['efficiency_metrics']:
                plt.subplot(2, 2, 4)
                fidelity = self.efficiency_results['efficiency_metrics']['reconstruction_fidelity']
                
                snr_spatial = fidelity.get('spatial_snr_db', 0)
                snr_angular = fidelity.get('angular_snr_db', 0)
                corr_spatial = fidelity.get('spatial_correlation', 0) * 100
                corr_angular = fidelity.get('angular_correlation', 0) * 100
                
                x = np.arange(2)
                width = 0.35
                
                plt.bar(x - width/2, [snr_spatial, snr_angular], width, label='SNR (dB)', color='lightblue')
                plt.bar(x + width/2, [corr_spatial, corr_angular], width, label='Correlation (%)', color='lightcoral')
                
                plt.xlabel('Channel Type')
                plt.ylabel('Value')
                plt.title('Reconstruction Fidelity')
                plt.xticks(x, ['Spatial', 'Angular'])
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Efficiency visualizations saved to {self.save_dir}")
            
        except Exception as e:
            print(f"⚠️ Could not generate efficiency visualizations: {str(e)}")


def test_maximum_efficiency_coding():
    """Test the Maximum Efficiency Coding hypothesis."""
    print("=" * 80)
    print("🎯 TESTING HYPOTHESIS 2: MAXIMUM EFFICIENCY CODING")
    print("=" * 80)
    
    # Generate test data
    print("📊 Generating test data...")
    torch.manual_seed(42)
    batch_size, channels, height, width = 1, 64, 8, 12
    
    spatial_latents = torch.randn(batch_size, channels, height, width)
    angular_latents = torch.randn(batch_size, channels, height, width)
    
    # Add structure for better compression
    spatial_latents[:, :32, :, :] *= 0.3  # Create low-energy channels
    angular_latents[:, :32, :, :] *= 0.3
    spatial_latents[:, :, 4:, 6:] *= 0.05  # Create sparse regions
    angular_latents[:, :, 4:, 6:] *= 0.05
    
    # Add some correlation structure for vector quantization
    for i in range(0, 64, 4):
        spatial_latents[:, i:i+2, :, :] = spatial_latents[:, i:i+1, :, :].repeat(1, 2, 1, 1) * torch.randn(1, 2, 1, 1) * 0.1
        angular_latents[:, i:i+2, :, :] = angular_latents[:, i:i+1, :, :].repeat(1, 2, 1, 1) * torch.randn(1, 2, 1, 1) * 0.1
    
    print(f"✅ Generated latents: {spatial_latents.shape}")
    
    # Initialize and test the hypothesis
    efficiency_coder = MaximumEfficiencyCoding(save_dir="results_hypothesis_2")
    
    # Train the pipeline
    efficiency_coder.train_pipeline(spatial_latents, angular_latents)
    
    # Test maximum efficiency compression
    efficiency_results = efficiency_coder.compress_with_efficiency_focus(spatial_latents, angular_latents)
    
    # Generate comprehensive report
    report = efficiency_coder.generate_efficiency_report(spatial_latents, angular_latents)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 MAXIMUM EFFICIENCY CODING RESULTS SUMMARY")
    print("=" * 80)
    
    if 'error' not in efficiency_results:
        overall_perf = efficiency_results.get('overall_performance', {})
        timing = efficiency_results.get('timing', {})
        
        print(f"🎯 Compression Ratio: {overall_perf.get('compression_ratio', 0):.2f}x")
        print(f"📉 Size Reduction: {overall_perf.get('size_reduction_percent', 0):.1f}%")
        print(f"💾 Bits per Element: {overall_perf.get('bits_per_element', 32):.2f}")
        print(f"⏱️  Encoding Time: {timing.get('encoding_time', 0):.2f}s")
        print(f"⏱️  Decoding Time: {timing.get('decoding_time', 0):.2f}s")
        print(f"🔍 Reconstruction Quality: {overall_perf.get('reconstruction_quality', 'unknown')}")
        
        # Component analysis
        if 'component_analysis' in efficiency_results:
            comp_analysis = efficiency_results['component_analysis']
            print(f"\n📊 Component Analysis:")
            
            if 'clipping_sparsification' in comp_analysis:
                sparsity = comp_analysis['clipping_sparsification']
                print(f"  ✂️  Sparsity: {sparsity.get('spatial_sparsity_ratio', 0)*100:.1f}% spatial, "
                      f"{sparsity.get('angular_sparsity_ratio', 0)*100:.1f}% angular")
            
            if 'transform_coding' in comp_analysis:
                transform = comp_analysis['transform_coding']
                print(f"  ⚡ Energy Compaction: {transform.get('energy_compaction_ratio', 0)*100:.1f}%")
            
            if 'vector_quantization' in comp_analysis:
                vq = comp_analysis['vector_quantization']
                vq_stats = vq.get('compression_stats', {})
                print(f"  🎨 Vector Quantization: {vq_stats.get('compression_ratio', 0):.2f}x compression")
            
            if 'bitstream_structuring' in comp_analysis:
                struct = comp_analysis['bitstream_structuring']
                print(f"  🗜️  Bitstream Structuring: {struct.get('rle_efficiency', 0):.2f}x RLE efficiency")
    else:
        print(f"❌ Compression failed: {efficiency_results['error']}")
    
    print(f"\n🎯 Pipeline Components: {len(efficiency_coder.pipeline_config)} configured")
    print(f"📁 Results saved to: {efficiency_coder.save_dir}")
    print(f"⏱️  Training time: {efficiency_coder.training_results['training_time']:.2f}s")
    
    print("\n✅ Maximum Efficiency Coding hypothesis test completed!")
    return efficiency_coder, report


if __name__ == "__main__":
    test_maximum_efficiency_coding() 
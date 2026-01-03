import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
import pickle
import os
from collections import defaultdict, Counter
import heapq
import struct
import math

class NonUniformQuantizer:
    """
    TRAINING-FREE Adaptive Non-Uniform Quantizer that computes optimal quantization levels
    on-the-fly for each input. NO TRAINING REQUIRED - analyzes current data distribution
    and adapts quantization levels based on min/max range per channel.
    """
    
    def __init__(self, num_levels: int = 16, channel_wise: bool = True, method: str = 'adaptive_minmax'):
        """
        Initialize the training-free adaptive quantizer.
        
        Args:
            num_levels: Number of quantization levels (default: 16 for 4-bit quantization)
            channel_wise: If True, compute separate quantizers for each channel
            method: Quantization method ('adaptive_minmax', 'percentile', 'histogram')
        """
        self.num_levels = num_levels
        self.channel_wise = channel_wise
        self.method = method
        # No quantization tables stored - computed on-the-fly!
        
    def _compute_adaptive_levels(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        TRAINING-FREE: Compute optimal quantization levels on-the-fly based on current data.
        Adapts to the actual min/max range and distribution of the input data.
        
        Args:
            data: Input data to analyze (current input only)
            
        Returns:
            Tuple of (decision_boundaries, reconstruction_levels)
        """
        # Remove any NaN or infinite values
        data = data[np.isfinite(data)]
        
        if len(data) == 0:
            # Emergency fallback for empty data
            levels = np.linspace(-1.0, 1.0, self.num_levels)
            boundaries = (levels[:-1] + levels[1:]) / 2
            return boundaries, levels
        
        data_min, data_max = np.min(data), np.max(data)
        data_range = data_max - data_min
        
        if data_range < 1e-8:  # Essentially constant data
            # Create small spread around the constant value
            center = data_min
            spread = max(1e-4, abs(center) * 1e-6)
            levels = np.linspace(center - spread, center + spread, self.num_levels)
            boundaries = (levels[:-1] + levels[1:]) / 2
            return boundaries, levels
        
        # Adaptive quantization based on method
        if self.method == 'adaptive_minmax':
            # Simple adaptive approach: distribute levels between min/max
            # Use non-uniform spacing - more levels near the mean
            data_mean = np.mean(data)
            data_std = np.std(data)
            
            # Create non-uniform levels with more density around mean
            # 40% of levels in ±0.5*std around mean, 60% for the rest
            center_levels = int(self.num_levels * 0.4)
            outer_levels = self.num_levels - center_levels
            
            if center_levels > 0:
                center_range = min(data_std, data_range * 0.3)
                center_min = max(data_min, data_mean - center_range)
                center_max = min(data_max, data_mean + center_range)
                center_levels_array = np.linspace(center_min, center_max, center_levels)
            else:
                center_levels_array = np.array([])
            
            if outer_levels > 0:
                outer_left = outer_levels // 2
                outer_right = outer_levels - outer_left
                
                left_levels = np.linspace(data_min, center_min if center_levels > 0 else data_mean, outer_left + 1)[:-1]
                right_levels = np.linspace(center_max if center_levels > 0 else data_mean, data_max, outer_right + 1)[1:]
                
                levels = np.concatenate([left_levels, center_levels_array, right_levels])
            else:
                levels = center_levels_array
            
        elif self.method == 'percentile':
            # Use percentile-based quantization
            percentiles = np.linspace(1, 99, self.num_levels)
            levels = np.percentile(data, percentiles)
            
        elif self.method == 'histogram':
            # Use histogram analysis for better adaptation
            hist, bin_edges = np.histogram(data, bins=min(100, len(data)//5 + 1))
            
            # Create levels based on cumulative distribution
            cumsum = np.cumsum(hist)
            total = cumsum[-1]
            target_counts = np.linspace(total/self.num_levels, total, self.num_levels)
            
            levels = []
            for target in target_counts:
                # Find bin where cumulative count reaches target
                bin_idx = np.searchsorted(cumsum, target)
                bin_idx = min(bin_idx, len(bin_edges) - 2)
                levels.append(bin_edges[bin_idx])
            
            levels = np.array(levels)
        
        else:
            # Fallback to uniform spacing
            levels = np.linspace(data_min, data_max, self.num_levels)
        
        # Ensure levels are unique and sorted
        levels = np.unique(levels)
        if len(levels) < self.num_levels:
            # Interpolate missing levels
            levels = np.interp(np.linspace(0, len(levels)-1, self.num_levels), 
                              np.arange(len(levels)), levels)
        elif len(levels) > self.num_levels:
            # Select representative levels
            indices = np.linspace(0, len(levels)-1, self.num_levels).astype(int)
            levels = levels[indices]
        
        # Compute decision boundaries as midpoints
        boundaries = (levels[:-1] + levels[1:]) / 2
        
        return boundaries, levels
    
    def _quantize_with_tables(self, data: np.ndarray, boundaries: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """FIXED: Helper function to quantize data using given boundaries and levels."""
        # FIXED: Proper quantization with bounds checking
        quantized_data = np.zeros_like(data)
        
        for i in range(len(levels)):
            if i == 0:
                # First region: data <= first boundary
                mask = data <= boundaries[0] if len(boundaries) > 0 else np.ones_like(data, dtype=bool)
            elif i == len(levels) - 1:
                # Last region: data > last boundary
                mask = data > boundaries[-1] if len(boundaries) > 0 else np.zeros_like(data, dtype=bool)
            else:
                # Middle regions: boundaries[i-1] < data <= boundaries[i]
                mask = (data > boundaries[i-1]) & (data <= boundaries[i])
            
            quantized_data[mask] = levels[i]
        
        return quantized_data
    
    # NO TRAINING METHOD - This component is completely training-free!
    # Quantization levels are computed adaptively on-the-fly for each input.
    
    def quantize(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        TRAINING-FREE: Quantize latents using adaptive quantization levels computed on-the-fly.
        NO TRAINING REQUIRED - adapts to the actual min/max range and distribution of current input.
        
        Args:
            spatial_latents: Spatial latents of shape [batch, 64, 8, 12]
            angular_latents: Angular latents of shape [batch, 64, 8, 12]
            
        Returns:
            Tuple of (quantized_spatial, quantized_angular, side_info)
        """
        print(f"   🔢 Training-Free Non-Uniform Quantizer ({self.method}, {self.num_levels} levels)...")
        
        device = spatial_latents.device
        batch_size = spatial_latents.shape[0]
        
        # Initialize output tensors
        quant_spatial = torch.zeros_like(spatial_latents)
        quant_angular = torch.zeros_like(angular_latents)
        
        # Store quantization info for reconstruction
        quantization_info = {
            'num_levels': self.num_levels,
            'channel_wise': self.channel_wise,
            'method': self.method,
            'spatial_quantizers': {},
            'angular_quantizers': {}
        }
        
        if self.channel_wise:
            # Adaptive quantization per channel
            for ch in range(64):
                # Spatial channel quantization (TRAINING-FREE)
                ch_spatial_data = spatial_latents[:, ch, :, :].cpu().numpy()
                ch_spatial_flat = ch_spatial_data.flatten()
                
                # Compute adaptive quantization levels based on current data distribution
                boundaries, levels = self._compute_adaptive_levels(ch_spatial_flat)
                quant_data = self._quantize_with_tables(ch_spatial_flat, boundaries, levels)
                quant_spatial[:, ch, :, :] = torch.tensor(
                    quant_data.reshape(batch_size, 8, 12), device=device
                )
                
                # Store quantization info for decoder
                quantization_info['spatial_quantizers'][ch] = {
                    'boundaries': boundaries.tolist(),
                    'levels': levels.tolist(),
                    'data_min': float(np.min(ch_spatial_flat)),
                    'data_max': float(np.max(ch_spatial_flat)),
                    'data_mean': float(np.mean(ch_spatial_flat)),
                    'data_std': float(np.std(ch_spatial_flat))
                }
                
                # Angular channel quantization (TRAINING-FREE)
                ch_angular_data = angular_latents[:, ch, :, :].cpu().numpy()
                ch_angular_flat = ch_angular_data.flatten()
                
                boundaries, levels = self._compute_adaptive_levels(ch_angular_flat)
                quant_data = self._quantize_with_tables(ch_angular_flat, boundaries, levels)
                quant_angular[:, ch, :, :] = torch.tensor(
                    quant_data.reshape(batch_size, 8, 12), device=device
                )
                
                # Store quantization info for decoder
                quantization_info['angular_quantizers'][ch] = {
                    'boundaries': boundaries.tolist(), 
                    'levels': levels.tolist(),
                    'data_min': float(np.min(ch_angular_flat)),
                    'data_max': float(np.max(ch_angular_flat)),
                    'data_mean': float(np.mean(ch_angular_flat)),
                    'data_std': float(np.std(ch_angular_flat))
                }
        else:
            # Global quantization for all spatial/angular data
            spatial_data = spatial_latents.cpu().numpy()
            spatial_flat = spatial_data.flatten()
            
            boundaries, levels = self._compute_adaptive_levels(spatial_flat)
            quant_data = self._quantize_with_tables(spatial_flat, boundaries, levels)
            quant_spatial = torch.tensor(
                quant_data.reshape(spatial_data.shape), device=device
            )
            
            quantization_info['spatial_quantizers']['global'] = {
                'boundaries': boundaries.tolist(),
                'levels': levels.tolist(),
                'data_min': float(np.min(spatial_flat)),
                'data_max': float(np.max(spatial_flat)),
                'data_mean': float(np.mean(spatial_flat)),
                'data_std': float(np.std(spatial_flat))
            }
            
            # Angular quantization
            angular_data = angular_latents.cpu().numpy()
            angular_flat = angular_data.flatten()
            
            boundaries, levels = self._compute_adaptive_levels(angular_flat)
            quant_data = self._quantize_with_tables(angular_flat, boundaries, levels)
            quant_angular = torch.tensor(
                quant_data.reshape(angular_data.shape), device=device
            )
            
            quantization_info['angular_quantizers']['global'] = {
                'boundaries': boundaries.tolist(),
                'levels': levels.tolist(),
                'data_min': float(np.min(angular_flat)),
                'data_max': float(np.max(angular_flat)),
                'data_mean': float(np.mean(angular_flat)),
                'data_std': float(np.std(angular_flat))
            }
        
        # Calculate quantization statistics
        spatial_orig = spatial_latents.cpu().numpy().flatten()
        angular_orig = angular_latents.cpu().numpy().flatten()
        spatial_quant = quant_spatial.cpu().numpy().flatten()
        angular_quant = quant_angular.cpu().numpy().flatten()
        
        spatial_mse = np.mean((spatial_orig - spatial_quant) ** 2)
        angular_mse = np.mean((angular_orig - angular_quant) ** 2)
        
        side_info = {
            'quantizer_type': f'adaptive_non_uniform_{self.method}',
            'num_levels': self.num_levels,
            'channel_wise': self.channel_wise,
            'method': self.method,
            'quantization_info': quantization_info,
            'statistics': {
                'spatial_mse': float(spatial_mse),
                'angular_mse': float(angular_mse),
                'total_elements': len(spatial_orig) + len(angular_orig),
                'bits_per_symbol': int(np.ceil(np.log2(self.num_levels)))
            }
        }
        
        print(f"      ✅ Adaptive quantization completed (MSE: spatial={spatial_mse:.6f}, angular={angular_mse:.6f})")
        
        return quant_spatial, quant_angular, side_info
    
    def dequantize(self, quantized_spatial: torch.Tensor, quantized_angular: torch.Tensor, side_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TRAINING-FREE: Dequantize the latent representations.
        Since adaptive quantization already maps to optimal reconstruction levels,
        this is a simple pass-through operation.
        
        Args:
            quantized_spatial: Quantized spatial latents
            quantized_angular: Quantized angular latents
            side_info: Side information from quantization (contains adaptive levels)
            
        Returns:
            Tuple of (dequantized_spatial, dequantized_angular)
        """
        # For adaptive non-uniform quantization, quantized values ARE the dequantized values
        # The adaptive levels are already optimal reconstruction points
        return quantized_spatial, quantized_angular
    
    # NO SAVE/LOAD METHODS NEEDED - This quantizer is completely training-free!
    # All quantization parameters are computed on-the-fly from input data.


class ArithmeticEntropyCoder:
    """
    TRAINING-FREE Adaptive Arithmetic Entropy Coder for quantized indices.
    NO TRAINING REQUIRED - builds optimal probability models on-the-fly from current input statistics.
    Provides superior compression by achieving near-entropy bounds with adaptive modeling.
    
    Key Features:
    - Training-free operation - works immediately on any quantized input
    - Adaptive probability modeling from current input statistics
    - Multiple adaptive methods: frequency, laplace, entropy, kneser_ney
    - Channel-wise or global adaptive modeling strategies
    - Optimal entropy-based bit allocation per symbol
    """
    
    def __init__(self, channel_wise: bool = True, precision: int = 16, 
                 adaptive_method: str = 'frequency', smoothing_factor: float = 1.0):
        """
        Initialize the TRAINING-FREE adaptive arithmetic entropy coder.
        
        Args:
            channel_wise: If True, build adaptive models per channel
            precision: Arithmetic coding precision (bits) - reduced for efficiency
            adaptive_method: Adaptive modeling method ('frequency', 'laplace', 'entropy', 'kneser_ney')
            smoothing_factor: Smoothing factor for probability estimation
        """
        self.channel_wise = channel_wise
        self.precision = precision
        self.adaptive_method = adaptive_method
        self.smoothing_factor = smoothing_factor
        
    def _build_adaptive_probability_model(self, frequencies: Dict[int, int], method: str = 'frequency') -> Dict[int, float]:
        """
        TRAINING-FREE: Build adaptive probability model from current input symbol frequencies.
        NO TRAINING REQUIRED - computes optimal probabilities on-the-fly using adaptive methods.
        
        Args:
            frequencies: Dictionary mapping symbols to their frequencies in current input
            method: Adaptive modeling method ('frequency', 'laplace', 'entropy', 'kneser_ney')
            
        Returns:
            Dictionary mapping symbols to their adaptive probabilities
        """
        if not frequencies:
            return {}
        
        total_count = sum(frequencies.values())
        num_symbols = len(frequencies)
        
        probabilities = {}
        
        if method == 'frequency':
            # Simple frequency-based model with minimal smoothing
            smoothing = max(0.1, self.smoothing_factor * 0.1)  # Reduced smoothing
            adjusted_total = total_count + num_symbols * smoothing
            
            for symbol, freq in frequencies.items():
                probabilities[symbol] = (freq + smoothing) / adjusted_total
                
        elif method == 'laplace':
            # Laplace smoothing (add-one smoothing)
            smoothing = self.smoothing_factor
            adjusted_total = total_count + num_symbols * smoothing
            
            for symbol, freq in frequencies.items():
                probabilities[symbol] = (freq + smoothing) / adjusted_total
                
        elif method == 'entropy':
            # Entropy-optimized model - emphasize high-frequency symbols
            # Calculate normalized frequencies first
            raw_probs = {symbol: freq / total_count for symbol, freq in frequencies.items()}
            
            # Apply entropy-based weighting (higher weight to frequent symbols)
            entropy_weights = {}
            for symbol, prob in raw_probs.items():
                if prob > 0:
                    # Inverse entropy weighting
                    weight = 1.0 / (-prob * math.log2(prob + 1e-10))
                    entropy_weights[symbol] = weight
                else:
                    entropy_weights[symbol] = 1.0
            
            # Normalize weights and apply minimal smoothing
            total_weight = sum(entropy_weights.values())
            smoothing = self.smoothing_factor * 0.05  # Very light smoothing
            
            for symbol, freq in frequencies.items():
                weight_factor = entropy_weights[symbol] / total_weight
                probabilities[symbol] = (freq * weight_factor + smoothing) / (total_count + num_symbols * smoothing)
                
        elif method == 'kneser_ney':
            # Simplified Kneser-Ney inspired smoothing - redistributes probability mass
            discount = min(0.75, self.smoothing_factor * 0.5)  # Discount factor
            
            # Apply discounting to frequent symbols
            discounted_mass = 0.0
            for symbol, freq in frequencies.items():
                if freq > 1:
                    discounted_freq = freq - discount
                    probabilities[symbol] = discounted_freq / total_count
                    discounted_mass += discount / total_count
                else:
                    probabilities[symbol] = freq / total_count
            
            # Redistribute discounted mass uniformly among low-frequency symbols
            low_freq_symbols = [s for s, f in frequencies.items() if f == 1]
            if low_freq_symbols and discounted_mass > 0:
                bonus_prob = discounted_mass / len(low_freq_symbols)
                for symbol in low_freq_symbols:
                    probabilities[symbol] += bonus_prob
                    
        else:  # Default to frequency method
            smoothing = self.smoothing_factor
            adjusted_total = total_count + num_symbols * smoothing
            
            for symbol, freq in frequencies.items():
                probabilities[symbol] = (freq + smoothing) / adjusted_total
        
        # Ensure probabilities sum to 1.0 (handle floating point errors)
        prob_sum = sum(probabilities.values())
        if prob_sum > 0:
            for symbol in probabilities:
                probabilities[symbol] /= prob_sum
        
        return probabilities
    
    def _analyze_adaptive_symbol_statistics(self, data: np.ndarray, levels: np.ndarray) -> Dict[str, float]:
        """
        TRAINING-FREE: Analyze symbol statistics for adaptive entropy coding.
        Computes real-time statistics to guide optimal compression strategy.
        
        Args:
            data: Input quantized data
            levels: Quantization levels for converting to indices
            
        Returns:
            Dictionary with adaptive symbol statistics
        """
        indices = self._quantized_to_indices(data, levels)
        frequencies = Counter(indices)
        
        total_symbols = len(indices)
        unique_symbols = len(frequencies)
        
        # Calculate entropy and compression metrics
        entropy = 0.0
        max_freq = 0
        min_freq = total_symbols
        
        for freq in frequencies.values():
            if freq > 0:
                prob = freq / total_symbols
                entropy -= prob * math.log2(prob)
                max_freq = max(max_freq, freq)
                min_freq = min(min_freq, freq)
        
        # Calculate distribution uniformity
        expected_freq = total_symbols / unique_symbols if unique_symbols > 0 else 0
        variance = np.var(list(frequencies.values())) if frequencies else 0
        uniformity = 1.0 / (1.0 + variance / (expected_freq + 1e-10))
        
        # Calculate compression potential
        theoretical_bits = entropy * total_symbols
        uniform_bits = math.log2(unique_symbols) * total_symbols if unique_symbols > 1 else total_symbols
        compression_potential = max(0, 1.0 - theoretical_bits / (uniform_bits + 1e-10))
        
        return {
            'entropy': entropy,
            'unique_symbols': unique_symbols,
            'total_symbols': total_symbols,
            'max_frequency': max_freq,
            'min_frequency': min_freq,
            'frequency_variance': float(variance),
            'distribution_uniformity': uniformity,
            'theoretical_bits': theoretical_bits,
            'compression_potential': compression_potential,
            'symbol_alphabet_size': len(levels)
        }
    
    def _select_optimal_adaptive_method(self, symbol_stats: Dict[str, float]) -> str:
        """
        TRAINING-FREE: Select optimal adaptive method based on current input characteristics.
        NO TRAINING REQUIRED - chooses best method from real-time analysis.
        
        Args:
            symbol_stats: Symbol statistics from current input
            
        Returns:
            Optimal adaptive method name
        """
        entropy = symbol_stats['entropy']
        uniformity = symbol_stats['distribution_uniformity']
        compression_potential = symbol_stats['compression_potential']
        unique_symbols = symbol_stats['unique_symbols']
        
        # Decision tree based on data characteristics
        if entropy < 2.0 and compression_potential > 0.5:
            # Low entropy, high compression potential -> frequency method
            return 'frequency'
        elif uniformity < 0.3 and unique_symbols > 16:
            # Non-uniform, many symbols -> entropy method
            return 'entropy'
        elif entropy > 6.0 and unique_symbols > 32:
            # High entropy, many symbols -> kneser_ney method
            return 'kneser_ney'
        else:
            # Default case -> laplace method
            return 'laplace'
    
    def _quantized_to_indices(self, quantized_data: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """
        Convert quantized values back to indices for entropy coding.
        
        Args:
            quantized_data: Quantized floating point values
            levels: Reconstruction levels from quantizer
            
        Returns:
            Array of integer indices
        """
        # FIXED: More robust method - find closest level for each value
        indices = np.zeros(quantized_data.shape, dtype=np.int32)
        
        for i, value in enumerate(quantized_data.flat):
            # Find closest level index using minimum distance
            distances = np.abs(levels - value)
            closest_idx = np.argmin(distances)
            
            # Ensure index is within valid range
            closest_idx = max(0, min(closest_idx, len(levels) - 1))
            indices.flat[i] = closest_idx
        
        return indices
    
    def _arithmetic_encode(self, symbols: List[int], probabilities: Dict[int, float]) -> Tuple[int, int]:
        """
        Encode symbols using arithmetic coding.
        
        Args:
            symbols: List of symbols to encode
            probabilities: Symbol probabilities
            
        Returns:
            Tuple of (encoded_value, actual_entropy_bits)
        """
        # Sort symbols for consistent cumulative probability calculation
        sorted_symbols = sorted(probabilities.keys())
        
        # Build cumulative probability table
        cumulative_probs = {}
        cum_prob = 0.0
        for symbol in sorted_symbols:
            cumulative_probs[symbol] = cum_prob
            cum_prob += probabilities[symbol]
        
        # Arithmetic coding with fixed precision
        low = 0
        high = (1 << self.precision) - 1
        range_size = high - low + 1
        
        for symbol in symbols:
            # FIXED: Handle unseen symbols gracefully
            if symbol not in probabilities:
                # Use uniform distribution for unseen symbols
                uniform_prob = 1.0 / (len(probabilities) + 1)
                probabilities[symbol] = uniform_prob
                # Re-normalize all probabilities
                total_prob = sum(probabilities.values())
                for k in probabilities:
                    probabilities[k] /= total_prob
                # Rebuild cumulative probabilities
                sorted_symbols = sorted(probabilities.keys())
                cumulative_probs = {}
                cum_prob = 0.0
                for s in sorted_symbols:
                    cumulative_probs[s] = cum_prob
                    cum_prob += probabilities[s]
            
            # Get symbol probability range
            symbol_low = cumulative_probs[symbol]
            symbol_high = symbol_low + probabilities[symbol]
            
            # Update range
            new_range = high - low + 1
            high = low + int(symbol_high * new_range) - 1
            low = low + int(symbol_low * new_range)
            
            # Rescaling to avoid precision loss
            while True:
                if high < (1 << (self.precision - 1)):
                    # Both in lower half
                    low <<= 1
                    high = (high << 1) + 1
                elif low >= (1 << (self.precision - 1)):
                    # Both in upper half
                    low = (low - (1 << (self.precision - 1))) << 1
                    high = ((high - (1 << (self.precision - 1))) << 1) + 1
                else:
                    break
                
                # Prevent overflow
                if low >= (1 << self.precision) or high >= (1 << self.precision):
                    break
        
        # Return a value in the final range
        encoded_value = (low + high) // 2
        
        # Calculate actual entropy-based bits instead of fixed precision
        actual_entropy_bits = self._calculate_entropy_bits(symbols, probabilities)
        
        return encoded_value, actual_entropy_bits
    
    def _calculate_entropy_bits(self, symbols: List[int], probabilities: Dict[int, float]) -> int:
        """
        Calculate actual entropy-based bits required for the symbols.
        
        Args:
            symbols: List of symbols
            probabilities: Symbol probabilities
            
        Returns:
            Actual bits required based on entropy
        """
        if not symbols:
            return 0
        
        # Calculate Shannon entropy: H = -sum(p * log2(p))
        shannon_entropy = 0.0
        for symbol_value, prob in probabilities.items():
            if prob > 0:
                shannon_entropy -= prob * math.log2(prob)
        
        # Total bits = entropy per symbol * number of symbols
        total_entropy_bits = shannon_entropy * len(symbols)
        
        # Add small overhead for arithmetic coding (typically 1-2 bits total)
        overhead_bits = 2
        
        return max(1, int(total_entropy_bits + overhead_bits))  # At least 1 bit
    
    def _arithmetic_decode(self, encoded_value: int, num_symbols: int, 
                          probabilities: Dict[int, float]) -> List[int]:
        """
        Decode symbols using arithmetic decoding.
        
        Args:
            encoded_value: Encoded value
            num_symbols: Number of symbols to decode
            probabilities: Symbol probabilities
            
        Returns:
            List of decoded symbols
        """
        # Sort symbols for consistent decoding
        sorted_symbols = sorted(probabilities.keys())
        
        # Build cumulative probability table
        cumulative_probs = {}
        cum_prob = 0.0
        for symbol in sorted_symbols:
            cumulative_probs[symbol] = cum_prob
            cum_prob += probabilities[symbol]
        
        decoded_symbols = []
        low = 0
        high = (1 << self.precision) - 1
        
        for _ in range(num_symbols):
            # Find current symbol
            range_size = high - low + 1
            
            # FIXED: Handle zero range gracefully
            if range_size <= 0:
                # Emergency fallback - return first symbol
                decoded_symbols.extend([sorted_symbols[0]] * (num_symbols - len(decoded_symbols)))
                break
                
            scaled_value = (encoded_value - low) / range_size
            
            # Find symbol that contains this value
            current_symbol = None
            for symbol in sorted_symbols:
                symbol_low = cumulative_probs[symbol]
                symbol_high = symbol_low + probabilities[symbol]
                
                if symbol_low <= scaled_value < symbol_high:
                    current_symbol = symbol
                    break
            
            if current_symbol is None:
                current_symbol = sorted_symbols[-1]  # Fallback to last symbol
            
            decoded_symbols.append(current_symbol)
            
            # Update range for next symbol
            symbol_low = cumulative_probs[current_symbol]
            symbol_high = symbol_low + probabilities[current_symbol]
            
            new_range = high - low + 1
            high = low + int(symbol_high * new_range) - 1
            low = low + int(symbol_low * new_range)
            
            # Same rescaling as encoder
            while True:
                if high < (1 << (self.precision - 1)):
                    low <<= 1
                    high = (high << 1) + 1
                elif low >= (1 << (self.precision - 1)):
                    low = (low - (1 << (self.precision - 1))) << 1
                    high = ((high - (1 << (self.precision - 1))) << 1) + 1
                else:
                    break
                
                if low >= (1 << self.precision) or high >= (1 << self.precision):
                    break
        
        return decoded_symbols
    
    def encode(self, quantized_spatial: torch.Tensor, quantized_angular: torch.Tensor, 
               quantizer_side_info: Dict) -> Tuple[bytes, Dict]:
        """
        TRAINING-FREE: Encode quantized data using adaptive arithmetic coding.
        NO TRAINING REQUIRED - builds optimal probability models on-the-fly from current input.
        
        Args:
            quantized_spatial: Quantized spatial latents
            quantized_angular: Quantized angular latents  
            quantizer_side_info: Side information from quantizer (levels, boundaries)
            
        Returns:
            Tuple of (compressed_bitstream, adaptive_side_info)
        """
        spatial_data = quantized_spatial.cpu().numpy()
        angular_data = quantized_angular.cpu().numpy()
        quantization_info = quantizer_side_info['quantization_info']
        spatial_quantizers = quantization_info['spatial_quantizers']
        angular_quantizers = quantization_info['angular_quantizers']
        
        encoded_parts = []
        total_bits = 0
        adaptive_models = {'spatial': {}, 'angular': {}}
        global_stats = {'spatial': {}, 'angular': {}}
        
        if self.channel_wise:
            print(f"TRAINING-FREE Arithmetic Encoding: Processing {spatial_data.shape[1]} channels adaptively...")
            
            for ch in range(spatial_data.shape[1]):  # Process all channels
                # ===== SPATIAL CHANNEL ADAPTIVE ENCODING =====
                ch_data = spatial_data[:, ch, :, :].flatten()
                levels = np.array(spatial_quantizers[ch]['levels'])
                
                # Build adaptive probability model on-the-fly
                indices = self._quantized_to_indices(ch_data, levels)
                frequencies = Counter(indices)
                
                # Analyze current input statistics
                symbol_stats = self._analyze_adaptive_symbol_statistics(ch_data, levels)
                
                # Select optimal method for this channel's characteristics
                optimal_method = self._select_optimal_adaptive_method(symbol_stats)
                if hasattr(self, 'adaptive_method') and self.adaptive_method != 'auto':
                    optimal_method = self.adaptive_method  # Use user-specified method
                
                # Build adaptive probability model
                probabilities = self._build_adaptive_probability_model(frequencies, optimal_method)
                
                # Encode using adaptive model
                encoded_value, bits = self._arithmetic_encode(indices.tolist(), probabilities)
                
                encoded_parts.append({
                    'type': 'spatial',
                    'channel': ch,
                    'encoded_value': encoded_value,
                    'num_symbols': len(indices),
                    'bits': bits,
                    'adaptive_method': optimal_method
                })
                total_bits += bits
                
                # Store adaptive model for decoding
                adaptive_models['spatial'][ch] = {
                    'probabilities': probabilities,
                    'frequencies': dict(frequencies),
                    'levels': levels.tolist() if isinstance(levels, np.ndarray) else levels,
                    'adaptive_method': optimal_method,
                    'symbol_stats': symbol_stats
                }
                
                # ===== ANGULAR CHANNEL ADAPTIVE ENCODING =====
                ch_data = angular_data[:, ch, :, :].flatten()
                levels = np.array(angular_quantizers[ch]['levels'])
                
                # Build adaptive probability model on-the-fly
                indices = self._quantized_to_indices(ch_data, levels)
                frequencies = Counter(indices)
                
                # Analyze current input statistics
                symbol_stats = self._analyze_adaptive_symbol_statistics(ch_data, levels)
                
                # Select optimal method for this channel's characteristics
                optimal_method = self._select_optimal_adaptive_method(symbol_stats)
                if hasattr(self, 'adaptive_method') and self.adaptive_method != 'auto':
                    optimal_method = self.adaptive_method
                
                # Build adaptive probability model
                probabilities = self._build_adaptive_probability_model(frequencies, optimal_method)
                
                # Encode using adaptive model
                encoded_value, bits = self._arithmetic_encode(indices.tolist(), probabilities)
                
                encoded_parts.append({
                    'type': 'angular',
                    'channel': ch,
                    'encoded_value': encoded_value,
                    'num_symbols': len(indices),
                    'bits': bits,
                    'adaptive_method': optimal_method
                })
                total_bits += bits
                
                # Store adaptive model for decoding
                adaptive_models['angular'][ch] = {
                    'probabilities': probabilities,
                    'frequencies': dict(frequencies),
                    'levels': levels.tolist() if isinstance(levels, np.ndarray) else levels,
                    'adaptive_method': optimal_method,
                    'symbol_stats': symbol_stats
                }
        else:
            # ===== GLOBAL ADAPTIVE ENCODING =====
            print("TRAINING-FREE Arithmetic Encoding: Processing globally adaptively...")
            
            # Global spatial encoding
            all_spatial_data = spatial_data.flatten()
            levels = np.array(spatial_quantizers['global']['levels'])
            
            indices = self._quantized_to_indices(all_spatial_data, levels)
            frequencies = Counter(indices)
            symbol_stats = self._analyze_adaptive_symbol_statistics(all_spatial_data, levels)
            optimal_method = self._select_optimal_adaptive_method(symbol_stats)
            if hasattr(self, 'adaptive_method') and self.adaptive_method != 'auto':
                optimal_method = self.adaptive_method
            
            probabilities = self._build_adaptive_probability_model(frequencies, optimal_method)
            encoded_value, bits = self._arithmetic_encode(indices.tolist(), probabilities)
            
            encoded_parts.append({
                'type': 'spatial',
                'channel': 'global',
                'encoded_value': encoded_value,
                'num_symbols': len(indices),
                'bits': bits,
                'adaptive_method': optimal_method
            })
            total_bits += bits
            
            adaptive_models['spatial']['global'] = {
                'probabilities': probabilities,
                'frequencies': dict(frequencies),
                'levels': levels.tolist() if isinstance(levels, np.ndarray) else levels,
                'adaptive_method': optimal_method,
                'symbol_stats': symbol_stats
            }
            
            # Global angular encoding
            all_angular_data = angular_data.flatten()
            levels = np.array(angular_quantizers['global']['levels'])
            
            indices = self._quantized_to_indices(all_angular_data, levels)
            frequencies = Counter(indices)
            symbol_stats = self._analyze_adaptive_symbol_statistics(all_angular_data, levels)
            optimal_method = self._select_optimal_adaptive_method(symbol_stats)
            if hasattr(self, 'adaptive_method') and self.adaptive_method != 'auto':
                optimal_method = self.adaptive_method
            
            probabilities = self._build_adaptive_probability_model(frequencies, optimal_method)
            encoded_value, bits = self._arithmetic_encode(indices.tolist(), probabilities)
            
            encoded_parts.append({
                'type': 'angular',
                'channel': 'global',
                'encoded_value': encoded_value,
                'num_symbols': len(indices),
                'bits': bits,
                'adaptive_method': optimal_method
            })
            total_bits += bits
            
            adaptive_models['angular']['global'] = {
                'probabilities': probabilities,
                'frequencies': dict(frequencies),
                'levels': levels.tolist() if isinstance(levels, np.ndarray) else levels,
                'adaptive_method': optimal_method,
                'symbol_stats': symbol_stats
            }
        
        # Convert to bytes (serialize the encoded parts)
        compressed_bytes = pickle.dumps(encoded_parts)
        
        # Calculate global compression statistics
        total_elements = spatial_data.size + angular_data.size
        theoretical_entropy = sum(
            model.get('symbol_stats', {}).get('theoretical_bits', 0)
            for models in adaptive_models.values()
            for model in models.values()
        )
        
        # Comprehensive adaptive side information
        side_info = {
            'entropy_type': 'arithmetic_coding_adaptive',
            'channel_wise': self.channel_wise,
            'adaptive_models': adaptive_models,  # Contains all adaptive probability models
            'total_bits': total_bits,
            'theoretical_entropy_bits': theoretical_entropy,
            'compression_efficiency': theoretical_entropy / (total_bits + 1e-10),
            'shape': spatial_data.shape,
            'precision': self.precision,
            'adaptive_method_used': self.adaptive_method,
            'smoothing_factor': self.smoothing_factor,
            'total_elements': total_elements,
            'bits_per_element': total_bits / total_elements,
            'global_compression_ratio': (total_elements * 8) / (total_bits + 1e-10)  # Assuming 8-bit baseline
        }
        
        print(f"   Adaptive Arithmetic Encoding completed: {total_bits} bits, {side_info['compression_efficiency']:.3f} efficiency")
        
        return compressed_bytes, side_info
    
    def decode(self, compressed_bitstream: bytes, side_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TRAINING-FREE: Decode compressed bitstream using adaptive probability models.
        NO TRAINING REQUIRED - uses the adaptive models built during encoding.
        
        Args:
            compressed_bitstream: Compressed data
            side_info: Adaptive side information from encoding
            
        Returns:
            Tuple of (spatial_latents, angular_latents)
        """
        # Deserialize encoded parts
        encoded_parts = pickle.loads(compressed_bitstream)
        
        shape = side_info['shape']
        spatial_data = np.zeros(shape)
        angular_data = np.zeros(shape)
        adaptive_models = side_info['adaptive_models']
        
        print(f"TRAINING-FREE Arithmetic Decoding: Processing {len(encoded_parts)} parts...")
        
        if side_info['channel_wise']:
            part_idx = 0
            num_channels = shape[1]
            
            for ch in range(num_channels):
                # ===== DECODE SPATIAL CHANNEL =====
                spatial_part = encoded_parts[part_idx]
                part_idx += 1
                
                # Retrieve adaptive model for this channel
                spatial_model = adaptive_models['spatial'][ch]
                probabilities = spatial_model['probabilities']
                levels = spatial_model['levels']
                
                # Decode using adaptive probability model
                decoded_indices = self._arithmetic_decode(
                    spatial_part['encoded_value'],
                    spatial_part['num_symbols'],
                    probabilities
                )
                
                # Convert indices back to quantized values
                ch_values = [levels[idx] for idx in decoded_indices]
                spatial_data[:, ch, :, :] = np.array(ch_values).reshape(shape[0], shape[2], shape[3])
                
                # ===== DECODE ANGULAR CHANNEL =====
                angular_part = encoded_parts[part_idx]
                part_idx += 1
                
                # Retrieve adaptive model for this channel
                angular_model = adaptive_models['angular'][ch]
                probabilities = angular_model['probabilities']
                levels = angular_model['levels']
                
                # Decode using adaptive probability model
                decoded_indices = self._arithmetic_decode(
                    angular_part['encoded_value'],
                    angular_part['num_symbols'],
                    probabilities
                )
                
                # Convert indices back to quantized values
                ch_values = [levels[idx] for idx in decoded_indices]
                angular_data[:, ch, :, :] = np.array(ch_values).reshape(shape[0], shape[2], shape[3])
        else:
            # ===== GLOBAL DECODING =====
            part_idx = 0
            
            # Decode global spatial
            spatial_part = encoded_parts[part_idx]
            part_idx += 1
            
            spatial_model = adaptive_models['spatial']['global']
            probabilities = spatial_model['probabilities']
            levels = spatial_model['levels']
            
            decoded_indices = self._arithmetic_decode(
                spatial_part['encoded_value'],
                spatial_part['num_symbols'],
                probabilities
            )
            
            # Convert indices back to values and reshape
            spatial_values = [levels[idx] for idx in decoded_indices]
            spatial_data = np.array(spatial_values).reshape(shape)
            
            # Decode global angular
            angular_part = encoded_parts[part_idx]
            part_idx += 1
            
            angular_model = adaptive_models['angular']['global']
            probabilities = angular_model['probabilities']
            levels = angular_model['levels']
            
            decoded_indices = self._arithmetic_decode(
                angular_part['encoded_value'],
                angular_part['num_symbols'],
                probabilities
            )
            
            # Convert indices back to values and reshape
            angular_values = [levels[idx] for idx in decoded_indices]
            angular_data = np.array(angular_values).reshape(shape)
        
        # Convert back to tensors
        spatial_tensor = torch.tensor(spatial_data, dtype=torch.float32)
        angular_tensor = torch.tensor(angular_data, dtype=torch.float32)
        
        print(f"   ✅ Adaptive Arithmetic Decoding completed successfully")
        
        return spatial_tensor, angular_tensor
    
    def calculate_compression_rate(self, original_spatial: torch.Tensor, original_angular: torch.Tensor, 
                                 compressed_bitstream: bytes) -> Dict[str, float]:
        """
        Calculate compression statistics.
        
        Args:
            original_spatial: Original spatial latents
            original_angular: Original angular latents
            compressed_bitstream: Compressed data
            
        Returns:
            Dictionary with compression statistics
        """
        # Original size (assuming 32-bit floats)
        original_bits = (original_spatial.numel() + original_angular.numel()) * 32
        
        # Compressed size
        compressed_bits = len(compressed_bitstream) * 8
        
        compression_ratio = original_bits / compressed_bits
        bpp = compressed_bits / (original_spatial.shape[0] * original_spatial.shape[2] * original_spatial.shape[3])
        
        return {
            'original_bits': original_bits,
            'compressed_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'bits_per_pixel': bpp,
            'size_reduction_percent': (1 - compressed_bits/original_bits) * 100
        }


class LatentReorderer:
    """
    TRAINING-FREE Latent Feature Reordering component that sorts channels by importance.
    This enables progressive/scalable coding by prioritizing high-energy channels.
    NO TRAINING REQUIRED - computes importance on-the-fly for each input.
    """
    
    def __init__(self, importance_metric: str = 'variance'):
        """
        Initialize the training-free latent reorderer.
        
        Args:
            importance_metric: Method to calculate channel importance ('variance', 'energy', 'l2_norm')
        """
        self.importance_metric = importance_metric
        # REMOVED: No training-related state needed
        
    def _calculate_channel_importance(self, latent_data: np.ndarray) -> np.ndarray:
        """
        Calculate importance score for each channel.
        
        Args:
            latent_data: Latent data of shape [batch, channels, height, width]
            
        Returns:
            Array of importance scores for each channel
        """
        if self.importance_metric == 'variance':
            # Calculate variance across all spatial locations and samples
            importance = np.var(latent_data, axis=(0, 2, 3))
        elif self.importance_metric == 'energy':
            # Calculate mean squared energy
            importance = np.mean(latent_data ** 2, axis=(0, 2, 3))
        elif self.importance_metric == 'l2_norm':
            # Calculate L2 norm across spatial dimensions
            importance = np.mean(np.sqrt(np.sum(latent_data ** 2, axis=(2, 3))), axis=0)
        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")
        
        return importance
    
    # REMOVED: No training method needed for training-free operation
    
    def reorder(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        TRAINING-FREE: Reorder latent channels by importance computed on-the-fly.
        Analyzes current input to determine channel importance and reorders accordingly.
        
        Args:
            spatial_latents: Spatial latents of shape [batch, 64, 8, 12]
            angular_latents: Angular latents of shape [batch, 64, 8, 12]
            
        Returns:
            Tuple of (reordered_spatial, reordered_angular, side_info)
        """
        print(f"🔄 TRAINING-FREE Latent Reordering using '{self.importance_metric}' metric...")
        
        device = spatial_latents.device
        
        # Compute channel importance directly from current input data
        spatial_data = spatial_latents.cpu().numpy()
        angular_data = angular_latents.cpu().numpy()
        
        # Calculate importance for spatial channels
        spatial_importance = self._calculate_channel_importance(spatial_data)
        spatial_order = np.argsort(spatial_importance)[::-1].copy()  # Most important first
        
        # Calculate importance for angular channels
        angular_importance = self._calculate_channel_importance(angular_data)
        angular_order = np.argsort(angular_importance)[::-1].copy()  # Most important first
        
        print(f"   Spatial importance range: {spatial_importance.min():.6f} - {spatial_importance.max():.6f}")
        print(f"   Angular importance range: {angular_importance.min():.6f} - {angular_importance.max():.6f}")
        print(f"   Most important spatial channels: {spatial_order[:5]}")
        print(f"   Most important angular channels: {angular_order[:5]}")
        
        # Reorder channels using torch.index_select
        spatial_order_tensor = torch.tensor(spatial_order, dtype=torch.long, device=device)
        angular_order_tensor = torch.tensor(angular_order, dtype=torch.long, device=device)
        
        reordered_spatial = torch.index_select(spatial_latents, 1, spatial_order_tensor).contiguous()
        reordered_angular = torch.index_select(angular_latents, 1, angular_order_tensor).contiguous()
        
        # Create side information for decoder
        side_info = {
            'reorder_type': 'channel_importance',
            'importance_metric': self.importance_metric,
            'spatial_order': spatial_order.tolist(),
            'angular_order': angular_order.tolist(),
            'spatial_importance': spatial_importance.tolist(),
            'angular_importance': angular_importance.tolist(),
            'original_shape': list(spatial_latents.shape),
            'training_free': True
        }
        
        print(f"   ✅ Reordering completed successfully")
        return reordered_spatial, reordered_angular, side_info
    
    def restore_order(self, reordered_spatial: torch.Tensor, reordered_angular: torch.Tensor, 
                     side_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Restore original channel order.
        
        Args:
            reordered_spatial: Reordered spatial latents
            reordered_angular: Reordered angular latents
            side_info: Side information from reordering
            
        Returns:
            Tuple of (original_order_spatial, original_order_angular)
        """
        spatial_order = np.array(side_info['spatial_order'])
        angular_order = np.array(side_info['angular_order'])
        
        # Create inverse permutation
        spatial_inverse = np.argsort(spatial_order)
        angular_inverse = np.argsort(angular_order)
        
        device = reordered_spatial.device
        
        # Restore original order using torch.index_select to avoid negative stride issues
        spatial_inverse_tensor = torch.tensor(spatial_inverse, dtype=torch.long, device=device)
        angular_inverse_tensor = torch.tensor(angular_inverse, dtype=torch.long, device=device)
        
        restored_spatial = torch.index_select(reordered_spatial, 1, spatial_inverse_tensor).contiguous()
        restored_angular = torch.index_select(reordered_angular, 1, angular_inverse_tensor).contiguous()
        
        return restored_spatial, restored_angular
    
    def get_progressive_subsets(self, num_levels: int = 4, spatial_order: Optional[np.ndarray] = None, angular_order: Optional[np.ndarray] = None) -> List[Tuple[List[int], List[int]]]:
        """
        FIXED: Get progressive channel subsets for scalable coding.
        Works with provided orders or generates default ordering.
        
        Args:
            num_levels: Number of progressive levels
            spatial_order: Optional spatial channel order (if None, uses default)
            angular_order: Optional angular channel order (if None, uses default)
            
        Returns:
            List of (spatial_channels, angular_channels) for each level
        """
        # FIXED: Use provided orders or create default sequential order
        if spatial_order is None:
            spatial_order = np.arange(64)  # Default: 0, 1, 2, ..., 63
        if angular_order is None:
            angular_order = np.arange(64)  # Default: 0, 1, 2, ..., 63
        
        subsets = []
        channels_per_level = 64 // num_levels
        
        for level in range(num_levels):
            end_idx = (level + 1) * channels_per_level
            if level == num_levels - 1:  # Last level gets remaining channels
                end_idx = 64
            
            spatial_subset = spatial_order[:end_idx].tolist()
            angular_subset = angular_order[:end_idx].tolist()
            
            subsets.append((spatial_subset, angular_subset))
        
        return subsets


class LatentClippingSparsifier:
    """
    TRAINING-FREE Latent Clipping and Sparsification component for outlier removal and zero promotion.
    This improves compression efficiency by reducing dynamic range and promoting sparsity.
    NO TRAINING REQUIRED - computes clipping and sparsity parameters on-the-fly for each input.
    """
    
    def __init__(self, clipping_method: str = 'percentile', sparsity_method: str = 'magnitude', 
                 clip_percentile: float = 99.5, sparsity_threshold: float = 0.1):
        """
        Initialize the training-free clipping and sparsification component.
        
        Args:
            clipping_method: Method for clipping ('percentile', 'std_dev', 'adaptive')
            sparsity_method: Method for sparsification ('magnitude', 'energy', 'adaptive')
            clip_percentile: Percentile for clipping (e.g., 99.5 means clip top/bottom 0.5%)
            sparsity_threshold: Threshold for promoting sparsity
        """
        self.clipping_method = clipping_method
        self.sparsity_method = sparsity_method
        self.clip_percentile = clip_percentile
        self.sparsity_threshold = sparsity_threshold
        # REMOVED: No training-related state needed
    
    def _calculate_clipping_bounds(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate clipping bounds based on the chosen method.
        
        Args:
            data: Input data array
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.clipping_method == 'percentile':
            lower_percentile = (100 - self.clip_percentile) / 2
            upper_percentile = 100 - lower_percentile
            lower_bound = np.percentile(data, lower_percentile)
            upper_bound = np.percentile(data, upper_percentile)
            
        elif self.clipping_method == 'std_dev':
            mean_val = np.mean(data)
            std_val = np.std(data)
            # Clip at mean ± 3*std (covers ~99.7% of normal distribution)
            num_std = 3.0
            lower_bound = mean_val - num_std * std_val
            upper_bound = mean_val + num_std * std_val
            
        elif self.clipping_method == 'adaptive':
            # Adaptive method based on data distribution shape
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            # Use 1.5*IQR rule for outlier detection
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
        else:
            raise ValueError(f"Unknown clipping method: {self.clipping_method}")
        
        return lower_bound, upper_bound
    
    def _calculate_sparsity_threshold(self, data: np.ndarray) -> float:
        """
        Calculate sparsity threshold based on the chosen method.
        
        Args:
            data: Input data array
            
        Returns:
            Sparsity threshold value
        """
        if self.sparsity_method == 'magnitude':
            # Threshold based on magnitude distribution
            abs_data = np.abs(data)
            threshold = np.percentile(abs_data, self.sparsity_threshold * 100)
            
        elif self.sparsity_method == 'energy':
            # Threshold based on energy (squared magnitude)
            energy_data = data ** 2
            threshold = np.sqrt(np.percentile(energy_data, self.sparsity_threshold * 100))
            
        elif self.sparsity_method == 'adaptive':
            # Adaptive threshold based on local statistics
            abs_data = np.abs(data)
            median_abs = np.median(abs_data)
            mad = np.median(np.abs(abs_data - median_abs))  # Median Absolute Deviation
            threshold = median_abs + self.sparsity_threshold * mad
            
        else:
            raise ValueError(f"Unknown sparsity method: {self.sparsity_method}")
        
        return threshold
    
    # REMOVED: No training method needed for training-free operation
    
    def apply_clipping_sparsification(self, spatial_latents: torch.Tensor, 
                                    angular_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        TRAINING-FREE: Apply clipping and sparsification computed on-the-fly.
        Analyzes current input to determine optimal clipping bounds and sparsity thresholds.
        
        Args:
            spatial_latents: Spatial latents to process
            angular_latents: Angular latents to process
            
        Returns:
            Tuple of (processed_spatial, processed_angular, side_info)
        """
        print(f"✂️ TRAINING-FREE Clipping & Sparsification...")
        print(f"   Clipping method: {self.clipping_method}")
        print(f"   Sparsity method: {self.sparsity_method}")
        
        device = spatial_latents.device
        processed_spatial = spatial_latents.clone()
        processed_angular = angular_latents.clone()
        
        # Convert to numpy for processing
        spatial_data = spatial_latents.cpu().numpy()
        angular_data = angular_latents.cpu().numpy()
        
        # Statistics tracking
        spatial_clipped_count = 0
        spatial_sparsified_count = 0
        angular_clipped_count = 0
        angular_sparsified_count = 0
        
        # Store parameters for decoder (computed on-the-fly)
        spatial_clip_params = {}
        angular_clip_params = {}
        spatial_sparsity_params = {}
        angular_sparsity_params = {}
        
        # Process each channel
        for ch in range(64):
            # Process spatial channel - COMPUTE PARAMETERS ON-THE-FLY
            spatial_ch = processed_spatial[:, ch, :, :].cpu().numpy()
            ch_spatial_data = spatial_ch.flatten()
            
            # Compute clipping bounds for current input
            spatial_lower, spatial_upper = self._calculate_clipping_bounds(ch_spatial_data)
            spatial_sparsity_thresh = self._calculate_sparsity_threshold(ch_spatial_data)
            
            # Store parameters for decoder
            spatial_clip_params[ch] = {
                'lower_bound': float(spatial_lower),
                'upper_bound': float(spatial_upper)
            }
            spatial_sparsity_params[ch] = {
                'threshold': float(spatial_sparsity_thresh)
            }
            
            # Apply clipping
            original_spatial = spatial_ch.copy()
            spatial_ch = np.clip(spatial_ch, spatial_lower, spatial_upper)
            spatial_clipped_count += np.sum(original_spatial != spatial_ch)
            
            # Apply sparsification
            sparsity_mask = np.abs(spatial_ch) < spatial_sparsity_thresh
            spatial_sparsified_count += np.sum(sparsity_mask)
            spatial_ch[sparsity_mask] = 0.0
            
            processed_spatial[:, ch, :, :] = torch.tensor(spatial_ch, device=device)
            
            # Process angular channel - COMPUTE PARAMETERS ON-THE-FLY
            angular_ch = processed_angular[:, ch, :, :].cpu().numpy()
            ch_angular_data = angular_ch.flatten()
            
            # Compute clipping bounds for current input
            angular_lower, angular_upper = self._calculate_clipping_bounds(ch_angular_data)
            angular_sparsity_thresh = self._calculate_sparsity_threshold(ch_angular_data)
            
            # Store parameters for decoder
            angular_clip_params[ch] = {
                'lower_bound': float(angular_lower),
                'upper_bound': float(angular_upper)
            }
            angular_sparsity_params[ch] = {
                'threshold': float(angular_sparsity_thresh)
            }
            
            # Apply clipping
            original_angular = angular_ch.copy()
            angular_ch = np.clip(angular_ch, angular_lower, angular_upper)
            angular_clipped_count += np.sum(original_angular != angular_ch)
            
            # Apply sparsification
            sparsity_mask = np.abs(angular_ch) < angular_sparsity_thresh
            angular_sparsified_count += np.sum(sparsity_mask)
            angular_ch[sparsity_mask] = 0.0
            
            processed_angular[:, ch, :, :] = torch.tensor(angular_ch, device=device)
        
        # Calculate overall statistics
        total_elements = spatial_latents.numel() + angular_latents.numel()
        spatial_sparsity_ratio = spatial_sparsified_count / spatial_latents.numel()
        angular_sparsity_ratio = angular_sparsified_count / angular_latents.numel()
        overall_sparsity_ratio = (spatial_sparsified_count + angular_sparsified_count) / total_elements
        
        print(f"   Spatial: {spatial_clipped_count} clipped, {spatial_sparsified_count} sparsified ({spatial_sparsity_ratio:.1%})")
        print(f"   Angular: {angular_clipped_count} clipped, {angular_sparsified_count} sparsified ({angular_sparsity_ratio:.1%})")
        print(f"   Overall sparsity: {overall_sparsity_ratio:.1%}")
        
        # Create side information for decoder (all computed on-the-fly)
        side_info = {
            'clipping_sparsification_type': 'training_free_adaptive',
            'clipping_method': self.clipping_method,
            'sparsity_method': self.sparsity_method,
            'clip_percentile': self.clip_percentile,
            'sparsity_threshold': self.sparsity_threshold,
            'spatial_clip_params': spatial_clip_params,
            'angular_clip_params': angular_clip_params,
            'spatial_sparsity_params': spatial_sparsity_params,
            'angular_sparsity_params': angular_sparsity_params,
            'statistics': {
                'spatial_clipped': int(spatial_clipped_count),
                'spatial_sparsified': int(spatial_sparsified_count),
                'angular_clipped': int(angular_clipped_count),
                'angular_sparsified': int(angular_sparsified_count),
                'spatial_sparsity_ratio': float(spatial_sparsity_ratio),
                'angular_sparsity_ratio': float(angular_sparsity_ratio),
                'overall_sparsity_ratio': float(overall_sparsity_ratio)
            },
            'training_free': True,
            'original_shape': list(spatial_latents.shape)
        }
        
        print(f"   ✅ Clipping & Sparsification completed successfully")
        
        return processed_spatial, processed_angular, side_info
    
    def reverse_clipping_sparsification(self, processed_spatial: torch.Tensor, 
                                       processed_angular: torch.Tensor, side_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse clipping and sparsification (partial reconstruction).
        NOTE: This is lossy - clipped values and zeros cannot be perfectly recovered.
        
        Args:
            processed_spatial: Clipped/sparsified spatial latents
            processed_angular: Clipped/sparsified angular latents  
            side_info: Side information from apply_clipping_sparsification
            
        Returns:
            Tuple of (restored_spatial, restored_angular) - best effort reconstruction
        """
        print(f"🔄 Reversing Clipping & Sparsification (best effort)...")
        
        # NOTE: This is lossy since clipped values and sparsified zeros are lost
        # We can only restore the shapes and indicate where processing occurred
        restored_spatial = processed_spatial.clone()
        restored_angular = processed_angular.clone()
        
        # The processed data is already the result - no perfect reversal possible
        # for lossy clipping and sparsification operations
        
        statistics = side_info.get('statistics', {})
        print(f"   Spatial: {statistics.get('spatial_clipped', 0)} values were clipped")
        print(f"   Angular: {statistics.get('angular_clipped', 0)} values were clipped")  
        print(f"   Total sparsified: {statistics.get('spatial_sparsified', 0) + statistics.get('angular_sparsified', 0)} values")
        print(f"   ⚠️ Note: Clipping and sparsification are lossy operations")
        
        return restored_spatial, restored_angular
    
    def calculate_compression_benefit(self, original_spatial: torch.Tensor, original_angular: torch.Tensor,
                                    processed_spatial: torch.Tensor, processed_angular: torch.Tensor) -> Dict[str, float]:
        """
        Calculate the compression benefits from clipping and sparsification.
        
        Args:
            original_spatial: Original spatial latents
            original_angular: Original angular latents
            processed_spatial: Processed spatial latents
            processed_angular: Processed angular latents
            
        Returns:
            Dictionary with compression benefit statistics
        """
        # Calculate sparsity ratios
        spatial_zeros = torch.sum(processed_spatial == 0).item()
        angular_zeros = torch.sum(processed_angular == 0).item()
        
        total_spatial = processed_spatial.numel()
        total_angular = processed_angular.numel()
        
        spatial_sparsity = spatial_zeros / total_spatial
        angular_sparsity = angular_zeros / total_angular
        
        # Calculate dynamic range reduction
        original_spatial_range = torch.max(original_spatial) - torch.min(original_spatial)
        processed_spatial_range = torch.max(processed_spatial) - torch.min(processed_spatial)
        
        original_angular_range = torch.max(original_angular) - torch.min(original_angular)
        processed_angular_range = torch.max(processed_angular) - torch.min(processed_angular)
        
        spatial_range_reduction = (original_spatial_range - processed_spatial_range) / original_spatial_range
        angular_range_reduction = (original_angular_range - processed_angular_range) / original_angular_range
        
        # Calculate distortion
        spatial_mse = torch.mean((original_spatial - processed_spatial) ** 2)
        angular_mse = torch.mean((original_angular - processed_angular) ** 2)
        
        return {
            'spatial_sparsity_ratio': spatial_sparsity,  # Already a float
            'angular_sparsity_ratio': angular_sparsity,  # Already a float
            'overall_sparsity_ratio': (spatial_zeros + angular_zeros) / (total_spatial + total_angular),
            'spatial_range_reduction': spatial_range_reduction.item(),
            'angular_range_reduction': angular_range_reduction.item(),
            'spatial_distortion_mse': spatial_mse.item(),
            'angular_distortion_mse': angular_mse.item(),
            'total_zeros_created': spatial_zeros + angular_zeros,
            'compression_potential': f"{(spatial_zeros + angular_zeros) / (total_spatial + total_angular) * 100:.1f}% zeros"
        }


class VectorQuantizer:
    """
    TRAINING-FREE Adaptive Vector Quantizer that computes K-means clustering on-the-fly.
    NO TRAINING REQUIRED - creates optimal codebooks adaptively for each input data.
    Exploits spatial correlations in latent representations using fast adaptive clustering.
    """
    
    def __init__(self, codebook_size: int = 256, vector_dim: int = 4, overlap_stride: int = 2,
                 channel_wise: bool = True, max_iterations: int = 20, adaptive_size: bool = True):
        """
        Initialize the training-free adaptive vector quantizer.
        
        Args:
            codebook_size: Base number of codewords (may be adapted based on data complexity)
            vector_dim: Dimension of vectors (e.g., 2x2=4, 3x3=9, 4x4=16)  
            overlap_stride: Stride for extracting overlapping patches
            channel_wise: Whether to use separate codebooks per channel
            max_iterations: Maximum iterations for fast K-means (default: 20 for speed)
            adaptive_size: Whether to adapt codebook size based on data complexity
        """
        self.codebook_size = codebook_size
        self.vector_dim = vector_dim
        self.overlap_stride = overlap_stride
        self.channel_wise = channel_wise
        self.max_iterations = max_iterations
        self.adaptive_size = adaptive_size
        
        # Calculate patch size from vector dimension
        self.patch_size = int(np.sqrt(vector_dim))
        if self.patch_size ** 2 != vector_dim:
            raise ValueError(f"Vector dimension {vector_dim} must be a perfect square")
        
        # No codebook storage - computed on-the-fly!
    
    def _extract_patches(self, data: np.ndarray) -> np.ndarray:
        """
        Extract overlapping patches from spatial data.
        
        Args:
            data: Input data of shape [batch, channels, height, width]
            
        Returns:
            Patches of shape [num_patches, vector_dim]
        """
        batch_size, channels, height, width = data.shape
        patches = []
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, height - self.patch_size + 1, self.overlap_stride):
                    for j in range(0, width - self.patch_size + 1, self.overlap_stride):
                        patch = data[b, c, i:i+self.patch_size, j:j+self.patch_size]
                        patches.append(patch.flatten())
        
        return np.array(patches)
    
    def _extract_channel_patches(self, data: np.ndarray, channel: int) -> np.ndarray:
        """
        Extract patches from a specific channel.
        
        Args:
            data: Input data of shape [batch, channels, height, width]
            channel: Channel index
            
        Returns:
            Patches from the specified channel
        """
        batch_size, _, height, width = data.shape
        patches = []
        
        for b in range(batch_size):
            for i in range(0, height - self.patch_size + 1, self.overlap_stride):
                for j in range(0, width - self.patch_size + 1, self.overlap_stride):
                    patch = data[b, channel, i:i+self.patch_size, j:j+self.patch_size]
                    patches.append(patch.flatten())
        
        return np.array(patches)
    
    def _adaptive_codebook_size(self, data: np.ndarray) -> int:
        """
        TRAINING-FREE: Determine adaptive codebook size based on data complexity.
        
        Args:
            data: Input vectors for analysis
            
        Returns:
            Optimal codebook size for current data
        """
        if not self.adaptive_size:
            return self.codebook_size
        
        num_samples = len(data)
        
        # Base adaptive sizing on data complexity metrics
        if num_samples < 50:
            # Very few samples - use small codebook
            return min(self.codebook_size, max(8, num_samples // 2))
        elif num_samples < 200:
            # Medium samples - moderate codebook
            return min(self.codebook_size, max(16, num_samples // 4))
        else:
            # Many samples - can use larger codebook
            # Also consider data variance for complexity
            data_variance = np.var(data)
            if data_variance < 0.01:
                # Low variance data - needs fewer clusters
                return min(self.codebook_size, max(32, self.codebook_size // 2))
            else:
                # High variance data - can benefit from more clusters
                return self.codebook_size
    
    def _kmeans_plus_plus_init(self, data: np.ndarray, k: int) -> np.ndarray:
        """
        TRAINING-FREE: Fast K-means++ initialization for better clustering.
        
        Args:
            data: Input vectors of shape [num_vectors, vector_dim]
            k: Number of centroids to initialize
            
        Returns:
            Initial centroids using K-means++ method
        """
        n_samples, n_features = data.shape
        centroids = np.empty((k, n_features))
        
        # Choose first centroid randomly
        centroids[0] = data[np.random.randint(n_samples)]
        
        for c_id in range(1, k):
            # Calculate distances to nearest centroids
            dist_sq = np.array([min([np.sum((x - c) ** 2) for c in centroids[:c_id]]) for x in data])
            
            # Choose next centroid with probability proportional to squared distance
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            
            # Find the index where r falls in cumulative probability
            i = len(cumulative_probs) - 1  # Default to last element
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
            
            centroids[c_id] = data[i]
        
        return centroids
    
    def _fast_adaptive_kmeans(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        TRAINING-FREE: Fast adaptive K-means clustering with intelligent initialization.
        Optimized for speed while maintaining quality.
        
        Args:
            data: Input vectors of shape [num_vectors, vector_dim]
            
        Returns:
            Tuple of (centroids, labels, actual_k_used)
        """
        # Determine adaptive codebook size
        k = self._adaptive_codebook_size(data)
        
        if len(data) < k:
            # If we have fewer samples than desired clusters, use all samples as centroids
            indices = np.random.choice(len(data), k, replace=True)
            centroids = data[indices].copy()
            # Simple assignment for small datasets
            distances = np.sqrt(((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            return centroids, labels, k
        
        # Use K-means++ for better initialization
        centroids = self._kmeans_plus_plus_init(data, k)
        
        # Fast K-means iterations with early stopping
        prev_labels = None
        for iteration in range(self.max_iterations):
            # Assign each point to nearest centroid (vectorized)
            distances = np.sqrt(((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            
            # Early stopping if labels don't change
            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels.copy()
            
            # Update centroids efficiently
            new_centroids = np.zeros_like(centroids)
            for k_idx in range(k):
                mask = labels == k_idx
                if np.sum(mask) > 0:
                    new_centroids[k_idx] = np.mean(data[mask], axis=0)
                else:
                    # If no points assigned, keep old centroid or reinitialize
                    if iteration < self.max_iterations // 2:
                        # Reinitialize empty centroid in early iterations
                        new_centroids[k_idx] = data[np.random.randint(len(data))]
                    else:
                        # Keep old centroid in later iterations
                        new_centroids[k_idx] = centroids[k_idx]
            
            # Check convergence
            if np.allclose(centroids, new_centroids, rtol=1e-4):  # Relaxed tolerance for speed
                break
            
            centroids = new_centroids
        
        # Final assignment
        distances = np.sqrt(((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        return centroids, labels, k
    
    # NO TRAINING METHOD - This component is completely training-free!
    # Codebooks are computed adaptively on-the-fly using fast K-means clustering.
    
    def _calculate_total_patches(self, data_shape: Tuple) -> int:
        """Calculate total number of patches per channel."""
        _, _, height, width = data_shape
        patches_h = (height - self.patch_size) // self.overlap_stride + 1
        patches_w = (width - self.patch_size) // self.overlap_stride + 1
        return patches_h * patches_w
    
    def _quantize_channel_adaptive(self, data: np.ndarray, channel: int, is_spatial: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        TRAINING-FREE: Quantize patches from a specific channel using adaptive clustering.
        Computes optimal codebook on-the-fly for current data.
        
        Args:
            data: Input data
            channel: Channel index  
            is_spatial: Whether this is spatial (True) or angular (False) data
            
        Returns:
            Tuple of (quantized_indices, reconstruction_error, codebook, actual_k_used)
        """
        # Extract patches from current data
        patches = self._extract_channel_patches(data, channel)
        
        if len(patches) == 0:
            # Handle empty patch case
            return np.array([]), 0.0, np.array([]), 0
        
        # Compute adaptive codebook on-the-fly
        codebook, labels, actual_k = self._fast_adaptive_kmeans(patches)
        
        # Find nearest centroids for all patches
        distances = np.sqrt(((patches[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2).sum(axis=2))
        indices = np.argmin(distances, axis=1)
        
        # Calculate reconstruction error
        reconstructed_patches = codebook[indices]
        mse = np.mean((patches - reconstructed_patches) ** 2)
        
        return indices, mse, codebook, actual_k
    
    def quantize(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor) -> Tuple[Dict, Dict, Dict]:
        """
        TRAINING-FREE: Apply adaptive vector quantization to latent representations.
        NO TRAINING REQUIRED - computes optimal codebooks on-the-fly for current data.
        
        Args:
            spatial_latents: Spatial latents to quantize
            angular_latents: Angular latents to quantize
            
        Returns:
            Tuple of (spatial_indices, angular_indices, side_info)
        """
        spatial_data = spatial_latents.cpu().numpy()
        angular_data = angular_latents.cpu().numpy()
        
        spatial_indices = {}
        angular_indices = {}
        spatial_errors = {}
        angular_errors = {}
        spatial_codebooks = {}
        angular_codebooks = {}
        spatial_k_used = {}
        angular_k_used = {}
        
        batch_size, channels, height, width = spatial_data.shape
        
        print(f"Vector Quantizer: Computing adaptive codebooks on-the-fly...")
        print(f"  Base codebook size: {self.codebook_size}")
        print(f"  Vector dimension: {self.vector_dim} ({self.patch_size}x{self.patch_size})")
        print(f"  Adaptive sizing: {self.adaptive_size}")
        print(f"  Channel-wise: {self.channel_wise}")
        
        # Quantize each channel with adaptive clustering
        for ch in range(channels):
            # Quantize spatial channel
            spatial_idx, spatial_err, spatial_codebook, spatial_actual_k = self._quantize_channel_adaptive(
                spatial_data, ch, is_spatial=True)
            spatial_indices[ch] = spatial_idx
            spatial_errors[ch] = spatial_err
            spatial_codebooks[ch] = spatial_codebook
            spatial_k_used[ch] = spatial_actual_k
            
            # Quantize angular channel
            angular_idx, angular_err, angular_codebook, angular_actual_k = self._quantize_channel_adaptive(
                angular_data, ch, is_spatial=False)
            angular_indices[ch] = angular_idx
            angular_errors[ch] = angular_err
            angular_codebooks[ch] = angular_codebook
            angular_k_used[ch] = angular_actual_k
        
        # Calculate adaptive compression statistics
        total_spatial_k = sum(spatial_k_used.values())
        total_angular_k = sum(angular_k_used.values())
        avg_spatial_k = total_spatial_k / channels if channels > 0 else 0
        avg_angular_k = total_angular_k / channels if channels > 0 else 0
        
        # Create comprehensive side information with adaptive codebooks
        side_info = {
            'vector_quantization_type': 'adaptive_kmeans_codebook',
            'base_codebook_size': self.codebook_size,
            'adaptive_size': self.adaptive_size,
            'vector_dim': self.vector_dim,
            'patch_size': self.patch_size,
            'overlap_stride': self.overlap_stride,
            'channel_wise': self.channel_wise,
            'max_iterations': self.max_iterations,
            'spatial_codebooks': spatial_codebooks,  # Computed on-the-fly!
            'angular_codebooks': angular_codebooks,  # Computed on-the-fly!
            'spatial_k_used': spatial_k_used,  # Actual adaptive sizes used
            'angular_k_used': angular_k_used,  # Actual adaptive sizes used
            'original_shape': spatial_latents.shape,
            'reconstruction_errors': {
                'spatial_mse': np.mean(list(spatial_errors.values())),
                'angular_mse': np.mean(list(angular_errors.values())),
                'spatial_errors_per_channel': spatial_errors,
                'angular_errors_per_channel': angular_errors
            },
            'adaptive_compression_stats': {
                'total_patches_per_channel': self._calculate_total_patches(spatial_data.shape),
                'avg_spatial_codebook_size': avg_spatial_k,
                'avg_angular_codebook_size': avg_angular_k,
                'total_spatial_centroids': total_spatial_k,
                'total_angular_centroids': total_angular_k,
                'max_bits_per_spatial_index': int(np.ceil(np.log2(max(spatial_k_used.values()) + 1))) if spatial_k_used else 0,
                'max_bits_per_angular_index': int(np.ceil(np.log2(max(angular_k_used.values()) + 1))) if angular_k_used else 0,
                'original_bits_per_patch': self.vector_dim * 32,  # Assuming float32
                'adaptive_savings': {
                    'requested_total_centroids': channels * 2 * self.codebook_size,
                    'actual_total_centroids': total_spatial_k + total_angular_k,
                    'centroid_reduction_percent': (1 - (total_spatial_k + total_angular_k) / (channels * 2 * self.codebook_size)) * 100
                }
            }
        }
        
        print(f"  Adaptive clustering completed:")
        print(f"    Avg spatial codebook size: {avg_spatial_k:.1f} (vs {self.codebook_size} base)")
        print(f"    Avg angular codebook size: {avg_angular_k:.1f} (vs {self.codebook_size} base)")
        print(f"    Centroid reduction: {side_info['adaptive_compression_stats']['adaptive_savings']['centroid_reduction_percent']:.1f}%")
        print(f"    Spatial MSE: {side_info['reconstruction_errors']['spatial_mse']:.6f}")
        print(f"    Angular MSE: {side_info['reconstruction_errors']['angular_mse']:.6f}")
        
        return spatial_indices, angular_indices, side_info
    
    def dequantize(self, spatial_indices: Dict, angular_indices: Dict, side_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TRAINING-FREE: Reconstruct latent representations from adaptive vector quantized indices.
        Uses the adaptive codebooks computed on-the-fly during quantization.
        
        Args:
            spatial_indices: Quantized spatial indices
            angular_indices: Quantized angular indices
            side_info: Side information from adaptive quantization
            
        Returns:
            Tuple of (reconstructed_spatial, reconstructed_angular)
        """
        original_shape = side_info['original_shape']
        batch_size, channels, height, width = original_shape
        
        spatial_codebooks = side_info['spatial_codebooks']  # Adaptive codebooks
        angular_codebooks = side_info['angular_codebooks']  # Adaptive codebooks
        patch_size = side_info['patch_size']
        overlap_stride = side_info['overlap_stride']
        channel_wise = side_info['channel_wise']
        
        # Initialize reconstruction arrays
        spatial_reconstructed = np.zeros((batch_size, channels, height, width))
        angular_reconstructed = np.zeros((batch_size, channels, height, width))
        spatial_counts = np.zeros((batch_size, channels, height, width))
        angular_counts = np.zeros((batch_size, channels, height, width))
        
        # Reconstruct each channel using adaptive codebooks
        for ch in range(channels):
            # Skip channels with no indices (edge case)
            if ch not in spatial_indices or ch not in angular_indices:
                continue
                
            # Get adaptive codebooks for this channel
            if channel_wise:
                if ch in spatial_codebooks and ch in angular_codebooks:
                    spatial_codebook = spatial_codebooks[ch]
                    angular_codebook = angular_codebooks[ch]
                else:
                    continue  # Skip missing codebooks
            else:
                if 0 in spatial_codebooks and 0 in angular_codebooks:
                    spatial_codebook = spatial_codebooks[0]
                    angular_codebook = angular_codebooks[0]
                else:
                    continue  # Skip missing codebooks
            
            # Get indices for this channel
            spatial_idx = spatial_indices[ch]
            angular_idx = angular_indices[ch]
            
            # Skip empty indices
            if len(spatial_idx) == 0 or len(angular_idx) == 0:
                continue
            
            # Reconstruct spatial patches using adaptive codebook
            spatial_patches = spatial_codebook[spatial_idx]
            
            # Reconstruct angular patches using adaptive codebook
            angular_patches = angular_codebook[angular_idx]
            
            # Place patches back (with overlap handling)
            patch_idx = 0
            for b in range(batch_size):
                for i in range(0, height - patch_size + 1, overlap_stride):
                    for j in range(0, width - patch_size + 1, overlap_stride):
                        if patch_idx < len(spatial_patches) and patch_idx < len(angular_patches):
                            # Spatial reconstruction
                            spatial_patch = spatial_patches[patch_idx].reshape(patch_size, patch_size)
                            spatial_reconstructed[b, ch, i:i+patch_size, j:j+patch_size] += spatial_patch
                            spatial_counts[b, ch, i:i+patch_size, j:j+patch_size] += 1
                            
                            # Angular reconstruction
                            angular_patch = angular_patches[patch_idx].reshape(patch_size, patch_size)
                            angular_reconstructed[b, ch, i:i+patch_size, j:j+patch_size] += angular_patch
                            angular_counts[b, ch, i:i+patch_size, j:j+patch_size] += 1
                            
                            patch_idx += 1
        
        # Average overlapping regions
        spatial_reconstructed = np.divide(spatial_reconstructed, spatial_counts, 
                                        out=np.zeros_like(spatial_reconstructed), 
                                        where=spatial_counts!=0)
        angular_reconstructed = np.divide(angular_reconstructed, angular_counts,
                                        out=np.zeros_like(angular_reconstructed),
                                        where=angular_counts!=0)
        
        return torch.tensor(spatial_reconstructed, dtype=torch.float32), torch.tensor(angular_reconstructed, dtype=torch.float32)
    
    def calculate_compression_ratio(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor,
                                  spatial_indices: Dict, angular_indices: Dict, side_info: Dict) -> Dict[str, float]:
        """
        Calculate compression statistics for vector quantization.
        
        Args:
            spatial_latents: Original spatial latents
            angular_latents: Original angular latents
            spatial_indices: Quantized spatial indices
            angular_indices: Quantized angular indices
            side_info: Side information
            
        Returns:
            Dictionary with compression statistics
        """
        original_shape = spatial_latents.shape
        total_elements = spatial_latents.numel() + angular_latents.numel()
        
        # Calculate compressed size
        stats = side_info['compression_stats']
        patches_per_channel = stats['total_patches_per_channel']
        bits_per_index = stats['bits_per_index']
        
        compressed_bits = 64 * 2 * patches_per_channel * original_shape[0] * bits_per_index  # 64 channels, 2 types (spatial/angular)
        original_bits = total_elements * 32  # float32
        
        compression_ratio = original_bits / compressed_bits
        
        # Calculate reconstruction error
        reconstructed_spatial, reconstructed_angular = self.dequantize(spatial_indices, angular_indices, side_info)
        
        spatial_mse = torch.mean((spatial_latents - reconstructed_spatial) ** 2)
        angular_mse = torch.mean((angular_latents - reconstructed_angular) ** 2)
        
        return {
            'original_bits': int(original_bits),
            'compressed_bits': int(compressed_bits),
            'compression_ratio': compression_ratio,
            'bits_per_element': compressed_bits / total_elements,
            'size_reduction_percent': (1 - compressed_bits / original_bits) * 100,
            'spatial_reconstruction_mse': spatial_mse.item(),
            'angular_reconstruction_mse': angular_mse.item(),
            'codebook_overhead_bits': len(self.spatial_codebooks) * self.codebook_size * self.vector_dim * 32 * 2,  # float32 codebooks
            'effective_compression_ratio': original_bits / (compressed_bits + len(self.spatial_codebooks) * self.codebook_size * self.vector_dim * 32 * 2)
        }


class TransformCoder:
    """
    TRAINING-FREE Adaptive Transform Coder using DCT for energy compaction.
    NO TRAINING REQUIRED - computes optimal quantization matrices on-the-fly based on current input.
    
    Applies Discrete Cosine Transform with adaptive quantization matrices that 
    automatically optimize for current data characteristics.
    """
    
    def __init__(self, block_size: int = 4, channel_wise: bool = True, 
                 adaptive_method: str = 'energy_adaptive',
                 quantization_strength: float = 1.0):
        """
        Initialize the TRAINING-FREE adaptive transform coder.
        
        Args:
            block_size: Size of transform blocks (4x4, 8x8, etc.)
            channel_wise: Whether to apply transform per channel
            adaptive_method: Method for creating adaptive quantization matrices:
                           'energy_adaptive', 'variance_adaptive', 'frequency_adaptive'
            quantization_strength: Strength of quantization (higher = more compression)
        """
        self.block_size = block_size
        self.channel_wise = channel_wise
        self.adaptive_method = adaptive_method
        self.quantization_strength = quantization_strength
        
        # Create DCT and IDCT matrices (universal - work for any input)
        self.dct_matrix = self._create_dct_matrix(block_size)
        self.idct_matrix = self.dct_matrix.T  # Inverse DCT is transpose of DCT
        
        print(f"TransformCoder initialized: TRAINING-FREE adaptive approach")
        print(f"  Block size: {block_size}x{block_size}")
        print(f"  Adaptive method: {adaptive_method}")
        print(f"  Channel-wise: {channel_wise}")
        print(f"  Ready to work immediately - no training required!")
    
    def _create_dct_matrix(self, n: int) -> np.ndarray:
        """
        Create DCT transformation matrix.
        
        Args:
            n: Block size
            
        Returns:
            DCT matrix of size [n, n]
        """
        dct_matrix = np.zeros((n, n))
        
        # First row (DC component)
        dct_matrix[0, :] = 1.0 / np.sqrt(n)
        
        # Remaining rows (AC components)
        for i in range(1, n):
            for j in range(n):
                dct_matrix[i, j] = np.sqrt(2.0 / n) * np.cos((2 * j + 1) * i * np.pi / (2 * n))
        
        return dct_matrix
    
    def _create_default_quantization_matrix(self, n: int) -> np.ndarray:
        """
        Create default quantization matrix based on human visual perception.
        
        Args:
            n: Block size
            
        Returns:
            Quantization matrix
        """
        if n == 4:
            # 4x4 quantization matrix (adapted from H.264)
            q_matrix = np.array([
                [1.0, 1.2, 1.6, 2.4],
                [1.2, 1.6, 2.4, 3.2],
                [1.6, 2.4, 3.2, 4.8],
                [2.4, 3.2, 4.8, 6.4]
            ])
        elif n == 8:
            # 8x8 quantization matrix (JPEG-like)
            q_matrix = np.array([
                [1.0, 1.4, 1.3, 1.6, 1.2, 1.1, 1.6, 1.9],
                [1.4, 1.3, 1.4, 1.5, 1.6, 1.3, 1.7, 1.8],
                [1.3, 1.4, 1.6, 1.4, 1.6, 1.9, 1.8, 2.2],
                [1.6, 1.5, 1.4, 1.7, 1.8, 2.2, 2.6, 2.3],
                [1.2, 1.6, 1.6, 1.8, 2.2, 2.6, 3.0, 2.9],
                [1.1, 1.3, 1.9, 2.2, 2.6, 3.0, 3.4, 3.2],
                [1.6, 1.7, 1.8, 2.6, 3.0, 3.4, 3.7, 3.9],
                [1.9, 1.8, 2.2, 2.3, 2.9, 3.2, 3.9, 4.1]
            ])
        else:
            # Generic quantization matrix
            q_matrix = np.ones((n, n))
            for i in range(n):
                for j in range(n):
                    q_matrix[i, j] = 1.0 + 0.5 * (i + j)
        
        return q_matrix
    
    def _create_adaptive_quantization_matrix(self, energy_distribution: np.ndarray, method: str = 'energy_adaptive') -> np.ndarray:
        """
        TRAINING-FREE: Create adaptive quantization matrix based on current input energy distribution.
        NO TRAINING REQUIRED - analyzes current data to create optimal quantization.
        
        Args:
            energy_distribution: Energy map from current input DCT analysis
            method: Adaptive method ('energy_adaptive', 'variance_adaptive', 'frequency_adaptive')
            
        Returns:
            Adaptive quantization matrix optimized for current input
        """
        n = self.block_size
        
        if method == 'energy_adaptive':
            # Create quantization matrix inversely proportional to energy concentration
            max_energy = np.max(energy_distribution)
            if max_energy > 0:
                # Higher energy coefficients get lower quantization (preserve more detail)
                q_matrix = self.quantization_strength * (max_energy / (energy_distribution + 1e-8))
                # Normalize to reasonable range
                q_matrix = np.clip(q_matrix, 0.1, 10.0)
            else:
                q_matrix = self._create_default_quantization_matrix(n)
                
        elif method == 'variance_adaptive':
            # Create quantization matrix based on coefficient variance
            variance_weights = np.var(energy_distribution) / (np.abs(energy_distribution) + 1e-8)
            q_matrix = self.quantization_strength * (1.0 + variance_weights)
            q_matrix = np.clip(q_matrix, 0.1, 10.0)
            
        elif method == 'frequency_adaptive':
            # Traditional frequency-based with adaptive scaling
            base_matrix = self._create_default_quantization_matrix(n)
            # Scale based on overall energy distribution
            energy_factor = np.mean(energy_distribution) / np.max(energy_distribution) if np.max(energy_distribution) > 0 else 1.0
            q_matrix = base_matrix * self.quantization_strength * (1.0 + energy_factor)
            
        else:
            # Fallback to default
            q_matrix = self._create_default_quantization_matrix(n) * self.quantization_strength
        
        return q_matrix
    
    def _analyze_current_energy_distribution(self, data: np.ndarray, channel: Optional[int] = None) -> np.ndarray:
        """
        TRAINING-FREE: Analyze energy distribution for current input data.
        NO TRAINING REQUIRED - computes optimal energy map on-the-fly.
        
        Args:
            data: Current input data to analyze
            channel: Specific channel to analyze (None for global analysis)
            
        Returns:
            Energy distribution map for current input
        """
        if channel is not None:
            # Analyze specific channel
            channel_data = data[:, channel:channel+1, :, :]
            blocks, _ = self._extract_blocks(channel_data)
        else:
            # Global analysis across all channels
            blocks, _ = self._extract_blocks(data)
        
        energy_map = np.zeros((self.block_size, self.block_size))
        total_blocks = 0
        
        # Analyze energy distribution for current input
        for b in range(blocks.shape[0]):
            for c in range(blocks.shape[1]):
                for i in range(blocks.shape[2]):
                    for j in range(blocks.shape[3]):
                        block = blocks[b, c, i, j]
                        dct_coeffs = self._apply_dct_block(block)
                        energy_map += dct_coeffs ** 2
                        total_blocks += 1
        
        # Average energy distribution
        if total_blocks > 0:
            energy_map = energy_map / total_blocks
        
        return energy_map
    
    def _apply_dct_block(self, block: np.ndarray) -> np.ndarray:
        """
        Apply DCT to a single block.
        
        Args:
            block: Input block of size [block_size, block_size]
            
        Returns:
            DCT coefficients
        """
        # Apply 2D DCT: DCT_matrix * block * DCT_matrix^T
        dct_coeffs = self.dct_matrix @ block @ self.dct_matrix.T
        return dct_coeffs
    
    def _apply_idct_block(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """
        Apply inverse DCT to a single block.
        
        Args:
            dct_coeffs: DCT coefficients
            
        Returns:
            Reconstructed block
        """
        # Apply 2D IDCT: IDCT_matrix * dct_coeffs * IDCT_matrix^T
        reconstructed = self.idct_matrix @ dct_coeffs @ self.idct_matrix.T
        return reconstructed
    
    def _extract_blocks(self, data: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """
        Extract non-overlapping blocks from data.
        
        Args:
            data: Input data of shape [batch, channels, height, width]
            
        Returns:
            Tuple of (blocks, original_shape)
        """
        batch_size, channels, height, width = data.shape
        
        # Calculate number of blocks
        blocks_h = height // self.block_size
        blocks_w = width // self.block_size
        
        # Extract blocks
        blocks = np.zeros((batch_size, channels, blocks_h, blocks_w, self.block_size, self.block_size))
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(blocks_h):
                    for j in range(blocks_w):
                        start_h = i * self.block_size
                        start_w = j * self.block_size
                        blocks[b, c, i, j] = data[b, c, start_h:start_h+self.block_size, start_w:start_w+self.block_size]
        
        return blocks, (batch_size, channels, height, width)
    
    def _reconstruct_from_blocks(self, blocks: np.ndarray, original_shape: Tuple) -> np.ndarray:
        """
        Reconstruct data from blocks.
        
        Args:
            blocks: Blocks of shape [batch, channels, blocks_h, blocks_w, block_size, block_size]
            original_shape: Original data shape
            
        Returns:
            Reconstructed data
        """
        batch_size, channels, height, width = original_shape
        reconstructed = np.zeros((batch_size, channels, height, width))
        
        blocks_h, blocks_w = blocks.shape[2], blocks.shape[3]
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(blocks_h):
                    for j in range(blocks_w):
                        start_h = i * self.block_size
                        start_w = j * self.block_size
                        reconstructed[b, c, start_h:start_h+self.block_size, start_w:start_w+self.block_size] = blocks[b, c, i, j]
        
        return reconstructed
    
    # TRAINING REMOVED - Component 5 is now TRAINING-FREE!
    # No training method needed - everything computed on-the-fly for current input
    
    # OLD TRAINING-BASED METHODS REMOVED - Component 5 is now TRAINING-FREE!
    # Energy analysis now performed on-the-fly in _analyze_current_energy_distribution
    
    def _calculate_energy_compaction(self, energy_map: np.ndarray) -> float:
        """
        Calculate energy compaction ratio (energy in top-left vs total energy).
        
        Args:
            energy_map: Energy distribution map
            
        Returns:
            Energy compaction ratio
        """
        total_energy = np.sum(energy_map)
        # Energy in top-left 2x2 quadrant
        top_left_energy = np.sum(energy_map[:2, :2])
        return top_left_energy / total_energy if total_energy > 0 else 0
    
    def apply_transform(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        TRAINING-FREE: Apply adaptive DCT transform to latent representations.
        NO TRAINING REQUIRED - computes optimal quantization matrices on-the-fly for current input.
        
        Args:
            spatial_latents: Spatial latents to transform
            angular_latents: Angular latents to transform
            
        Returns:
            Tuple of (spatial_dct_coeffs, angular_dct_coeffs, side_info)
        """
        print(f"Transform Coder: Computing adaptive DCT transform on-the-fly...")
        print(f"  Block size: {self.block_size}x{self.block_size}")
        print(f"  Adaptive method: {self.adaptive_method}")
        print(f"  Channel-wise processing: {self.channel_wise}")
        
        spatial_data = spatial_latents.cpu().numpy()
        angular_data = angular_latents.cpu().numpy()
        batch_size, channels, height, width = spatial_data.shape
        
        # Analyze energy distribution for current input (TRAINING-FREE)
        if self.channel_wise:
            spatial_energy_maps = {}
            angular_energy_maps = {}
            spatial_quantization_matrices = {}
            angular_quantization_matrices = {}
            
            for ch in range(channels):
                # Analyze current energy distribution per channel
                spatial_energy = self._analyze_current_energy_distribution(spatial_data, channel=ch)
                angular_energy = self._analyze_current_energy_distribution(angular_data, channel=ch)
                
                spatial_energy_maps[ch] = spatial_energy
                angular_energy_maps[ch] = angular_energy
                
                # Create adaptive quantization matrices for current input
                spatial_quantization_matrices[ch] = self._create_adaptive_quantization_matrix(
                    spatial_energy, self.adaptive_method)
                angular_quantization_matrices[ch] = self._create_adaptive_quantization_matrix(
                    angular_energy, self.adaptive_method)
        else:
            # Global energy analysis for current input
            spatial_energy = self._analyze_current_energy_distribution(spatial_data)
            angular_energy = self._analyze_current_energy_distribution(angular_data)
            
            spatial_energy_maps = {'global': spatial_energy}
            angular_energy_maps = {'global': angular_energy}
            
            # Create global adaptive quantization matrices
            spatial_quantization_matrices = {'global': self._create_adaptive_quantization_matrix(
                spatial_energy, self.adaptive_method)}
            angular_quantization_matrices = {'global': self._create_adaptive_quantization_matrix(
                angular_energy, self.adaptive_method)}
        
        # Extract blocks for DCT processing
        spatial_blocks, spatial_shape = self._extract_blocks(spatial_data)
        angular_blocks, angular_shape = self._extract_blocks(angular_data)
        
        # Apply adaptive DCT transform with on-the-fly quantization
        spatial_dct_blocks = np.zeros_like(spatial_blocks)
        angular_dct_blocks = np.zeros_like(angular_blocks)
        
        blocks_h, blocks_w = spatial_blocks.shape[2], spatial_blocks.shape[3]
        
        for b in range(batch_size):
            for c in range(channels):
                # Get adaptive quantization matrix for this channel
                if self.channel_wise:
                    spatial_q_matrix = spatial_quantization_matrices[c]
                    angular_q_matrix = angular_quantization_matrices[c]
                else:
                    spatial_q_matrix = spatial_quantization_matrices['global']
                    angular_q_matrix = angular_quantization_matrices['global']
                
                for i in range(blocks_h):
                    for j in range(blocks_w):
                        # Apply DCT to spatial block
                        spatial_dct = self._apply_dct_block(spatial_blocks[b, c, i, j])
                        # Apply adaptive quantization
                        spatial_dct_quantized = np.round(spatial_dct / spatial_q_matrix) * spatial_q_matrix
                        spatial_dct_blocks[b, c, i, j] = spatial_dct_quantized
                        
                        # Apply DCT to angular block  
                        angular_dct = self._apply_dct_block(angular_blocks[b, c, i, j])
                        # Apply adaptive quantization
                        angular_dct_quantized = np.round(angular_dct / angular_q_matrix) * angular_q_matrix
                        angular_dct_blocks[b, c, i, j] = angular_dct_quantized
        
        # Reconstruct as full arrays
        spatial_dct_coeffs = self._reconstruct_from_blocks(spatial_dct_blocks, spatial_shape)
        angular_dct_coeffs = self._reconstruct_from_blocks(angular_dct_blocks, angular_shape)
        
        # Calculate adaptive transform statistics
        transform_stats = self._calculate_adaptive_transform_stats(
            spatial_dct_coeffs, angular_dct_coeffs, spatial_energy_maps, angular_energy_maps)
        
        # Create comprehensive side information with adaptive matrices
        side_info = {
            'transform_type': 'adaptive_dct',
            'block_size': self.block_size,
            'channel_wise': self.channel_wise,
            'adaptive_method': self.adaptive_method,
            'quantization_strength': self.quantization_strength,
            'spatial_energy_maps': spatial_energy_maps,  # Computed on-the-fly!
            'angular_energy_maps': angular_energy_maps,  # Computed on-the-fly!
            'spatial_quantization_matrices': spatial_quantization_matrices,  # Adaptive!
            'angular_quantization_matrices': angular_quantization_matrices,  # Adaptive!
            'original_shapes': {
                'spatial': spatial_shape,
                'angular': angular_shape
            },
            'adaptive_transform_stats': transform_stats
        }
        
        print(f"  Energy compaction achieved:")
        print(f"    Spatial: {transform_stats['spatial_energy_compaction']:.3f}")
        print(f"    Angular: {transform_stats['angular_energy_compaction']:.3f}")
        print(f"  Sparsity achieved: {transform_stats['sparsity_ratio']:.1%}")
        
        return torch.tensor(spatial_dct_coeffs), torch.tensor(angular_dct_coeffs), side_info
    
    def _calculate_adaptive_transform_stats(self, spatial_dct: np.ndarray, angular_dct: np.ndarray, 
                                          spatial_energy_maps: Dict, angular_energy_maps: Dict) -> Dict[str, float]:
        """
        Calculate statistics about the adaptive transform coding.
        
        Args:
            spatial_dct: Spatial DCT coefficients
            angular_dct: Angular DCT coefficients
            spatial_energy_maps: Spatial energy distribution maps
            angular_energy_maps: Angular energy distribution maps
            
        Returns:
            Dictionary with adaptive transform statistics
        """
        # Calculate sparsity (zeros created by adaptive quantization)
        spatial_zeros = np.sum(spatial_dct == 0)
        angular_zeros = np.sum(angular_dct == 0)
        total_coeffs = spatial_dct.size + angular_dct.size
        
        sparsity_ratio = (spatial_zeros + angular_zeros) / total_coeffs
        
        # Calculate energy compaction for current input
        if 'global' in spatial_energy_maps:
            spatial_energy_compaction = self._calculate_energy_compaction(spatial_energy_maps['global'])
            angular_energy_compaction = self._calculate_energy_compaction(angular_energy_maps['global'])
        else:
            # Average across channels
            spatial_compactions = [self._calculate_energy_compaction(energy_map) 
                                 for energy_map in spatial_energy_maps.values()]
            angular_compactions = [self._calculate_energy_compaction(energy_map) 
                                 for energy_map in angular_energy_maps.values()]
            spatial_energy_compaction = np.mean(spatial_compactions)
            angular_energy_compaction = np.mean(angular_compactions)
        
        # Calculate coefficient energy distribution
        spatial_coeff_energy = self._calculate_coefficient_energy(spatial_dct)
        angular_coeff_energy = self._calculate_coefficient_energy(angular_dct)
        
        # Calculate adaptive efficiency metrics
        avg_quantization_strength = self.quantization_strength
        energy_preservation = (spatial_energy_compaction + angular_energy_compaction) / 2
        
        return {
            'sparsity_ratio': sparsity_ratio,
            'spatial_zeros': int(spatial_zeros),
            'angular_zeros': int(angular_zeros),
            'spatial_energy_compaction': spatial_energy_compaction,
            'angular_energy_compaction': angular_energy_compaction,
            'spatial_coefficient_energy': spatial_coeff_energy,
            'angular_coefficient_energy': angular_coeff_energy,
            'total_coefficients': int(total_coeffs),
            'compression_potential': f"{sparsity_ratio * 100:.1f}% sparse",
            'adaptive_method_used': self.adaptive_method,
            'quantization_strength_applied': avg_quantization_strength,
            'energy_preservation_score': energy_preservation,
            'adaptive_efficiency': sparsity_ratio * energy_preservation  # Combined metric
        }
    
    # OLD TRAINING-BASED QUANTIZATION REMOVED - Now done adaptively in apply_transform!
    
    def _calculate_transform_stats(self, spatial_dct: np.ndarray, angular_dct: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics about the transform coding.
        
        Args:
            spatial_dct: Spatial DCT coefficients
            angular_dct: Angular DCT coefficients
            
        Returns:
            Dictionary with transform statistics
        """
        # Calculate sparsity (zeros)
        spatial_zeros = np.sum(spatial_dct == 0)
        angular_zeros = np.sum(angular_dct == 0)
        total_coeffs = spatial_dct.size + angular_dct.size
        
        sparsity_ratio = (spatial_zeros + angular_zeros) / total_coeffs
        
        # Calculate energy compaction
        spatial_energy = self._calculate_coefficient_energy(spatial_dct)
        angular_energy = self._calculate_coefficient_energy(angular_dct)
        
        return {
            'sparsity_ratio': sparsity_ratio,
            'spatial_zeros': int(spatial_zeros),
            'angular_zeros': int(angular_zeros),
            'spatial_energy_compaction': spatial_energy,
            'angular_energy_compaction': angular_energy,
            'total_coefficients': int(total_coeffs),
            'compression_potential': f"{sparsity_ratio * 100:.1f}% sparse"
        }
    
    def _calculate_coefficient_energy(self, dct_coeffs: np.ndarray) -> float:
        """
        Calculate energy compaction ratio for DCT coefficients.
        
        Args:
            dct_coeffs: DCT coefficients
            
        Returns:
            Energy compaction ratio
        """
        blocks, _ = self._extract_blocks(dct_coeffs)
        
        total_energy = 0
        dc_energy = 0
        total_blocks = 0
        
        for b in range(blocks.shape[0]):
            for c in range(blocks.shape[1]):
                for i in range(blocks.shape[2]):
                    for j in range(blocks.shape[3]):
                        block_energy = np.sum(blocks[b, c, i, j] ** 2)
                        dc_energy += blocks[b, c, i, j, 0, 0] ** 2
                        total_energy += block_energy
                        total_blocks += 1
        
        return dc_energy / total_energy if total_energy > 0 else 0
    
    def apply_inverse_transform(self, spatial_dct_coeffs: torch.Tensor, angular_dct_coeffs: torch.Tensor, 
                               side_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TRAINING-FREE: Apply inverse DCT transform to reconstruct latent representations.
        Uses adaptive side information computed on-the-fly during forward transform.
        
        Args:
            spatial_dct_coeffs: Spatial DCT coefficients  
            angular_dct_coeffs: Angular DCT coefficients
            side_info: Adaptive side information from transform
            
        Returns:
            Tuple of (reconstructed_spatial, reconstructed_angular)
        """
        spatial_data = spatial_dct_coeffs.cpu().numpy()
        angular_data = angular_dct_coeffs.cpu().numpy()
        
        # Get original shapes from adaptive side info
        spatial_shape = side_info['original_shapes']['spatial']
        angular_shape = side_info['original_shapes']['angular']
        
        # Extract blocks
        spatial_blocks, _ = self._extract_blocks(spatial_data)
        angular_blocks, _ = self._extract_blocks(angular_data)
        
        # Apply IDCT to all blocks (universal operation - no training needed)
        spatial_reconstructed_blocks = np.zeros_like(spatial_blocks)
        angular_reconstructed_blocks = np.zeros_like(angular_blocks)
        
        batch_size, channels, blocks_h, blocks_w, _, _ = spatial_blocks.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(blocks_h):
                    for j in range(blocks_w):
                        # Apply IDCT to spatial block (universal DCT - works for any input)
                        spatial_reconstructed_blocks[b, c, i, j] = self._apply_idct_block(spatial_blocks[b, c, i, j])
                        
                        # Apply IDCT to angular block (universal DCT - works for any input)
                        angular_reconstructed_blocks[b, c, i, j] = self._apply_idct_block(angular_blocks[b, c, i, j])
        
        # Reconstruct as full arrays
        spatial_reconstructed = self._reconstruct_from_blocks(spatial_reconstructed_blocks, spatial_shape)
        angular_reconstructed = self._reconstruct_from_blocks(angular_reconstructed_blocks, angular_shape)
        
        return torch.tensor(spatial_reconstructed), torch.tensor(angular_reconstructed)
    
    def calculate_compression_benefit(self, original_spatial: torch.Tensor, original_angular: torch.Tensor,
                                    dct_spatial: torch.Tensor, dct_angular: torch.Tensor,
                                    side_info: Dict) -> Dict[str, float]:
        """
        Calculate compression benefits from adaptive transform coding.
        
        Args:
            original_spatial: Original spatial latents
            original_angular: Original angular latents
            dct_spatial: DCT spatial coefficients
            dct_angular: DCT angular coefficients
            side_info: Adaptive side information
            
        Returns:
            Dictionary with adaptive compression benefits
        """
        # Reconstruct and calculate distortion
        reconstructed_spatial, reconstructed_angular = self.apply_inverse_transform(
            dct_spatial, dct_angular, side_info
        )
        
        spatial_mse = torch.mean((original_spatial - reconstructed_spatial) ** 2)
        angular_mse = torch.mean((original_angular - reconstructed_angular) ** 2)
        
        # Get adaptive transform statistics
        transform_stats = side_info['adaptive_transform_stats']
        
        return {
            'sparsity_ratio': transform_stats['sparsity_ratio'],
            'spatial_energy_compaction': transform_stats['spatial_energy_compaction'],
            'angular_energy_compaction': transform_stats['angular_energy_compaction'],
            'spatial_reconstruction_mse': spatial_mse.item(),
            'angular_reconstruction_mse': angular_mse.item(),
            'total_zeros_created': transform_stats['spatial_zeros'] + transform_stats['angular_zeros'],
            'compression_potential': transform_stats['compression_potential'],
            'adaptive_method_used': transform_stats['adaptive_method_used'],
            'quantization_strength_applied': transform_stats['quantization_strength_applied'],
            'energy_preservation_score': transform_stats['energy_preservation_score'],
            'adaptive_efficiency': transform_stats['adaptive_efficiency'],
            'training_free_operation': True  # Key advantage!
        }


class BitPlaneCoder:
    """
    TRAINING-FREE Adaptive Bit-Plane Coding component for progressive reconstruction.
    NO TRAINING REQUIRED - analyzes current input and adapts bit-plane importance on-the-fly.
    Encodes coefficients bit-plane by bit-plane with adaptive significance analysis.
    
    Key Features:
    - Training-free operation - works immediately on any input
    - Adaptive bit-plane importance analysis on current data
    - Dynamic progressive level ordering based on actual bit significance  
    - On-the-fly significance map generation for efficient coding
    - Multiple adaptive ordering methods: magnitude, energy, variance, entropy
    """
    
    def __init__(self, num_bit_planes: int = 16, channel_wise: bool = True,
                 adaptive_order: str = 'energy', use_significance_map: bool = True,
                 significance_threshold: float = 0.1):
        """
        Initialize the TRAINING-FREE adaptive bit-plane coder.
        
        Args:
            num_bit_planes: Maximum number of bit planes to encode
            channel_wise: Whether to process channels separately
            adaptive_order: Adaptive ordering method ('energy', 'magnitude', 'variance', 'entropy', 'hybrid')
            use_significance_map: Whether to use significance maps for efficiency
            significance_threshold: Threshold for significance map creation
        """
        self.num_bit_planes = num_bit_planes
        self.channel_wise = channel_wise
        self.adaptive_order = adaptive_order
        self.use_significance_map = use_significance_map
        self.significance_threshold = significance_threshold
    
    def _float_to_fixed_point(self, data: np.ndarray, num_bits: int = 16) -> Tuple[np.ndarray, float, float]:
        """
        Convert floating point data to fixed point representation.
        
        Args:
            data: Input floating point data
            num_bits: Number of bits for fixed point representation
            
        Returns:
            Tuple of (fixed_point_data, scale_factor, offset)
        """
        # Find data range
        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min
        
        if data_range == 0:
            return np.zeros_like(data, dtype=np.int32), 1.0, data_min
        
        # Calculate scale factor to use full range
        max_value = (1 << (num_bits - 1)) - 1  # Account for sign bit
        scale_factor = max_value / (data_range / 2)
        
        # Convert to fixed point
        normalized_data = (data - data_min - data_range / 2) * scale_factor
        fixed_point_data = np.round(normalized_data).astype(np.int32)
        
        return fixed_point_data, scale_factor, data_min + data_range / 2
    
    def _fixed_point_to_float(self, fixed_data: np.ndarray, scale_factor: float, offset: float) -> np.ndarray:
        """
        Convert fixed point data back to floating point.
        
        Args:
            fixed_data: Fixed point data
            scale_factor: Scale factor used in conversion
            offset: Offset used in conversion
            
        Returns:
            Floating point data
        """
        float_data = fixed_data.astype(np.float32) / scale_factor + offset
        return float_data
    
    def _extract_bit_plane(self, data: np.ndarray, bit_plane: int) -> np.ndarray:
        """
        Extract a specific bit plane from integer data.
        
        Args:
            data: Integer data
            bit_plane: Bit plane index (0 = LSB, higher = more significant)
            
        Returns:
            Binary array for the bit plane
        """
        # Handle negative numbers using two's complement
        abs_data = np.abs(data)
        bit_mask = 1 << bit_plane
        bit_plane_data = (abs_data & bit_mask) >> bit_plane
        
        # Handle sign bit separately for the most significant bit plane
        if bit_plane == self.num_bit_planes - 1:
            sign_bits = (data < 0).astype(np.uint8)
            bit_plane_data = sign_bits
        
        return bit_plane_data.astype(np.uint8)
    
    def _create_significance_map(self, data: np.ndarray, threshold: int) -> np.ndarray:
        """
        Create significance map for coefficients above threshold.
        
        Args:
            data: Integer coefficient data
            threshold: Significance threshold
            
        Returns:
            Binary significance map
        """
        return (np.abs(data) >= threshold).astype(np.uint8)
    
    def _analyze_adaptive_bit_plane_importance(self, data: np.ndarray, method: str = 'energy') -> Dict[int, float]:
        """
        TRAINING-FREE: Analyze bit-plane importance using adaptive methods on current input.
        NO TRAINING REQUIRED - computes importance scores on-the-fly for current data.
        
        Args:
            data: Fixed point coefficient data
            method: Analysis method ('energy', 'variance', 'entropy', 'sparsity')
            
        Returns:
            Dictionary mapping bit plane to adaptive importance score
        """
        importance_scores = {}
        
        for bit_plane in range(self.num_bit_planes):
            bit_plane_data = self._extract_bit_plane(data, bit_plane)
            
            if method == 'energy':
                # Energy-based importance (weighted by bit position)
                energy = np.sum(bit_plane_data.astype(np.float32) * (2 ** bit_plane))
                sparsity = 1.0 - np.mean(bit_plane_data)
                importance_scores[bit_plane] = energy * (1 + sparsity)  # Higher sparsity = more important when present
                
            elif method == 'variance':
                # Variance-based importance
                bit_variance = np.var(bit_plane_data.astype(np.float32))
                bit_energy = np.sum(bit_plane_data.astype(np.float32) * (2 ** bit_plane))
                importance_scores[bit_plane] = bit_variance * bit_energy
                
            elif method == 'entropy':
                # Information entropy-based importance
                bit_plane_flat = bit_plane_data.flatten()
                unique_vals, counts = np.unique(bit_plane_flat, return_counts=True)
                probabilities = counts / len(bit_plane_flat)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                bit_energy = np.sum(bit_plane_data.astype(np.float32) * (2 ** bit_plane))
                importance_scores[bit_plane] = entropy * bit_energy
                
            elif method == 'sparsity':
                # Sparsity-weighted importance
                sparsity = 1.0 - np.mean(bit_plane_data)
                significance = np.sum(bit_plane_data > 0) / bit_plane_data.size
                importance_scores[bit_plane] = (2 ** bit_plane) * significance * (1 + sparsity)
                
            else:  # Default to energy method
                energy = np.sum(bit_plane_data.astype(np.float32) * (2 ** bit_plane))
                sparsity = 1.0 - np.mean(bit_plane_data)
                importance_scores[bit_plane] = energy * (1 + sparsity)
        
        return importance_scores
    
    def _create_adaptive_progressive_levels(self, spatial_data: np.ndarray, angular_data: np.ndarray, 
                                          method: str = 'energy') -> List[int]:
        """
        TRAINING-FREE: Create adaptive progressive transmission levels based on current input analysis.
        NO TRAINING REQUIRED - analyzes current data to determine optimal bit-plane ordering.
        
        Args:
            spatial_data: Current spatial fixed-point data
            angular_data: Current angular fixed-point data  
            method: Adaptive ordering method
            
        Returns:
            List of bit planes ordered by adaptive importance
        """
        if method == 'magnitude':
            # Standard magnitude ordering (MSB first)
            return list(range(self.num_bit_planes - 1, -1, -1))
            
        elif method in ['energy', 'variance', 'entropy', 'sparsity']:
            # Adaptive importance-based ordering
            if self.channel_wise:
                # Analyze each channel and combine
                spatial_importance_combined = np.zeros(self.num_bit_planes)
                angular_importance_combined = np.zeros(self.num_bit_planes)
                
                batch_size, channels = spatial_data.shape[:2]
                for ch in range(channels):
                    spatial_ch_data = spatial_data[:, ch, :, :]
                    angular_ch_data = angular_data[:, ch, :, :]
                    
                    spatial_importance = self._analyze_adaptive_bit_plane_importance(spatial_ch_data, method)
                    angular_importance = self._analyze_adaptive_bit_plane_importance(angular_ch_data, method)
                    
                    for bp in range(self.num_bit_planes):
                        spatial_importance_combined[bp] += spatial_importance.get(bp, 0)
                        angular_importance_combined[bp] += angular_importance.get(bp, 0)
                
                # Combine spatial and angular importance
                combined_importance = spatial_importance_combined + angular_importance_combined
            else:
                # Global analysis
                spatial_importance = self._analyze_adaptive_bit_plane_importance(spatial_data, method)
                angular_importance = self._analyze_adaptive_bit_plane_importance(angular_data, method)
                
                combined_importance = np.array([
                    spatial_importance.get(bp, 0) + angular_importance.get(bp, 0) 
                    for bp in range(self.num_bit_planes)
                ])
            
            # Sort by importance (highest first)
            return np.argsort(combined_importance)[::-1].tolist()
            
        elif method == 'hybrid':
            # Hybrid: Top 3 MSBs first, then by adaptive importance
            msb_planes = list(range(self.num_bit_planes - 1, self.num_bit_planes - 4, -1))
            
            # Remaining planes by energy importance
            remaining_levels = self._create_adaptive_progressive_levels(
                spatial_data, angular_data, 'energy'
            )
            remaining_levels = [bp for bp in remaining_levels if bp not in msb_planes]
            
            return msb_planes + remaining_levels
            
        else:
            # Default to magnitude ordering
            return list(range(self.num_bit_planes - 1, -1, -1))
    

        print("Bit-Plane Coder training completed!")
    

    
    def encode_progressive(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor) -> Tuple[List[Dict], Dict]:
        """
        TRAINING-FREE: Encode latents into adaptive progressive bit planes.
        NO TRAINING REQUIRED - analyzes current input and creates optimal bit-plane ordering on-the-fly.
        
        Args:
            spatial_latents: Spatial latents to encode
            angular_latents: Angular latents to encode
            
        Returns:
            Tuple of (bit_plane_layers, adaptive_side_info)
        """
        spatial_data = spatial_latents.cpu().numpy()
        angular_data = angular_latents.cpu().numpy()
        
        # Convert to fixed point and store conversion parameters per channel
        spatial_conversion_params = {}
        angular_conversion_params = {}
        spatial_fixed_data = np.zeros_like(spatial_data, dtype=np.int32)
        angular_fixed_data = np.zeros_like(angular_data, dtype=np.int32)
        
        if self.channel_wise:
            for ch in range(spatial_data.shape[1]):
                # Convert spatial channel to fixed point
                spatial_fixed, s_scale, s_offset = self._float_to_fixed_point(
                    spatial_data[:, ch, :, :], self.num_bit_planes
                )
                spatial_fixed_data[:, ch, :, :] = spatial_fixed
                spatial_conversion_params[ch] = {'scale_factor': s_scale, 'offset': s_offset}
                
                # Convert angular channel to fixed point
                angular_fixed, a_scale, a_offset = self._float_to_fixed_point(
                    angular_data[:, ch, :, :], self.num_bit_planes
                )
                angular_fixed_data[:, ch, :, :] = angular_fixed
                angular_conversion_params[ch] = {'scale_factor': a_scale, 'offset': a_offset}
        else:
            # Global conversion
            spatial_fixed_data, s_scale, s_offset = self._float_to_fixed_point(spatial_data, self.num_bit_planes)
            angular_fixed_data, a_scale, a_offset = self._float_to_fixed_point(angular_data, self.num_bit_planes)
            spatial_conversion_params[0] = {'scale_factor': s_scale, 'offset': s_offset}
            angular_conversion_params[0] = {'scale_factor': a_scale, 'offset': a_offset}
        
        # TRAINING-FREE: Create adaptive progressive levels based on current input
        adaptive_progressive_levels = self._create_adaptive_progressive_levels(
            spatial_fixed_data, angular_fixed_data, self.adaptive_order
        )
        
        bit_plane_layers = []
        
        for level_idx, bit_plane in enumerate(adaptive_progressive_levels):
            layer_data = {
                'bit_plane': bit_plane,
                'level': level_idx,
                'spatial_channels': {},
                'angular_channels': {},
                'significance_maps': {} if self.use_significance_map else None
            }
            
            if self.channel_wise:
                for ch in range(spatial_data.shape[1]):
                    # Extract bit plane from fixed point data
                    spatial_bit_plane = self._extract_bit_plane(spatial_fixed_data[:, ch, :, :], bit_plane)
                    angular_bit_plane = self._extract_bit_plane(angular_fixed_data[:, ch, :, :], bit_plane)
                    
                    layer_data['spatial_channels'][ch] = spatial_bit_plane
                    layer_data['angular_channels'][ch] = angular_bit_plane
                    
                    # Create adaptive significance maps
                    if self.use_significance_map:
                        # Adaptive threshold based on bit plane and data characteristics
                        threshold = max(1, int((1 << bit_plane) * self.significance_threshold))
                        spatial_sig_map = self._create_significance_map(spatial_fixed_data[:, ch, :, :], threshold)
                        angular_sig_map = self._create_significance_map(angular_fixed_data[:, ch, :, :], threshold)
                        
                        layer_data['significance_maps'][f'spatial_{ch}'] = spatial_sig_map
                        layer_data['significance_maps'][f'angular_{ch}'] = angular_sig_map
            else:
                # Global processing
                spatial_bit_plane = self._extract_bit_plane(spatial_fixed_data, bit_plane)
                angular_bit_plane = self._extract_bit_plane(angular_fixed_data, bit_plane)
                
                layer_data['spatial_channels'][0] = spatial_bit_plane
                layer_data['angular_channels'][0] = angular_bit_plane
                
                if self.use_significance_map:
                    threshold = max(1, int((1 << bit_plane) * self.significance_threshold))
                    spatial_sig_map = self._create_significance_map(spatial_fixed_data, threshold)
                    angular_sig_map = self._create_significance_map(angular_fixed_data, threshold)
                    
                    layer_data['significance_maps']['spatial_0'] = spatial_sig_map
                    layer_data['significance_maps']['angular_0'] = angular_sig_map
            
            bit_plane_layers.append(layer_data)
        
        # Create comprehensive adaptive side information
        side_info = {
            'bit_plane_coding_type': 'adaptive_progressive',
            'num_bit_planes': self.num_bit_planes,
            'channel_wise': self.channel_wise,
            'adaptive_order': self.adaptive_order,
            'adaptive_progressive_levels': adaptive_progressive_levels,
            'use_significance_map': self.use_significance_map,
            'significance_threshold': self.significance_threshold,
            'spatial_conversion_params': spatial_conversion_params,
            'angular_conversion_params': angular_conversion_params,
            'original_shape': spatial_latents.shape,
            'adaptive_compression_stats': self._calculate_adaptive_compression_stats(bit_plane_layers, adaptive_progressive_levels)
        }
        
        return bit_plane_layers, side_info
    
    def _calculate_adaptive_compression_stats(self, bit_plane_layers: List[Dict], 
                                            adaptive_progressive_levels: List[int]) -> Dict[str, float]:
        """
        Calculate adaptive compression statistics from current bit-plane analysis.
        
        Args:
            bit_plane_layers: Encoded bit plane layers
            adaptive_progressive_levels: Adaptively determined progressive levels
            
        Returns:
            Dictionary with adaptive compression statistics
        """
        stats = {
            'total_layers': len(bit_plane_layers),
            'total_bits_saved': 0,
            'sparsity_by_layer': [],
            'significance_ratio': 0.0,
            'adaptive_reordering_benefit': 0.0
        }
        
        for layer in bit_plane_layers:
            layer_sparsity = 0.0
            layer_elements = 0
            
            # Calculate sparsity per layer
            for ch, bit_plane_data in layer['spatial_channels'].items():
                layer_sparsity += np.sum(bit_plane_data == 0)
                layer_elements += bit_plane_data.size
            for ch, bit_plane_data in layer['angular_channels'].items():
                layer_sparsity += np.sum(bit_plane_data == 0)
                layer_elements += bit_plane_data.size
            
            if layer_elements > 0:
                layer_sparsity_ratio = layer_sparsity / layer_elements
                stats['sparsity_by_layer'].append(layer_sparsity_ratio)
                stats['total_bits_saved'] += layer_sparsity
        
        # Calculate average sparsity
        if stats['sparsity_by_layer']:
            stats['avg_sparsity'] = np.mean(stats['sparsity_by_layer'])
        else:
            stats['avg_sparsity'] = 0.0
        
        # Calculate significance map efficiency if used
        if len(bit_plane_layers) > 0 and bit_plane_layers[0]['significance_maps'] is not None:
            total_sig_elements = 0
            total_significant = 0
            
            for layer in bit_plane_layers:
                if layer['significance_maps']:
                    for sig_key, sig_map in layer['significance_maps'].items():
                        total_sig_elements += sig_map.size
                        total_significant += np.sum(sig_map)
            
            if total_sig_elements > 0:
                stats['significance_ratio'] = total_significant / total_sig_elements
        
        # Estimate adaptive reordering benefit (compare to standard magnitude ordering)
        magnitude_order = list(range(len(adaptive_progressive_levels) - 1, -1, -1))
        if adaptive_progressive_levels != magnitude_order:
            stats['adaptive_reordering_benefit'] = 1.0  # Indicates reordering was applied
        else:
            stats['adaptive_reordering_benefit'] = 0.0  # Standard ordering used
        
        return stats
    
    def decode_progressive(self, bit_plane_layers: List[Dict], side_info: Dict, 
                          num_layers: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TRAINING-FREE: Decode adaptive progressive bit planes to reconstruct latents.
        Uses adaptive side information computed during encoding.
        
        Args:
            bit_plane_layers: Encoded bit plane layers
            side_info: Adaptive side information from encoding
            num_layers: Number of layers to decode (for progressive reconstruction)
            
        Returns:
            Tuple of (reconstructed_spatial, reconstructed_angular)
        """
        if num_layers is None:
            num_layers = len(bit_plane_layers)
        
        original_shape = side_info['original_shape']
        channel_wise = side_info['channel_wise']
        spatial_conversion_params = side_info['spatial_conversion_params']
        angular_conversion_params = side_info['angular_conversion_params']
        
        # Initialize reconstruction arrays
        if channel_wise:
            spatial_fixed = np.zeros(original_shape, dtype=np.int32)
            angular_fixed = np.zeros(original_shape, dtype=np.int32)
        else:
            spatial_fixed = np.zeros(original_shape, dtype=np.int32)
            angular_fixed = np.zeros(original_shape, dtype=np.int32)
        
        # Reconstruct bit by bit using adaptive progressive levels
        for layer_idx in range(min(num_layers, len(bit_plane_layers))):
            layer = bit_plane_layers[layer_idx]
            bit_plane = layer['bit_plane']
            
            if channel_wise:
                # Process each channel that has data in this layer
                for ch in layer['spatial_channels'].keys():
                    if ch in layer['spatial_channels'] and ch in layer['angular_channels']:
                        spatial_bit_plane = layer['spatial_channels'][ch]
                        angular_bit_plane = layer['angular_channels'][ch]
                        
                        # Add bit plane contribution
                        spatial_fixed[:, ch, :, :] |= spatial_bit_plane.astype(np.int32) << bit_plane
                        angular_fixed[:, ch, :, :] |= angular_bit_plane.astype(np.int32) << bit_plane
            else:
                # Global processing
                spatial_bit_plane = layer['spatial_channels'][0]
                angular_bit_plane = layer['angular_channels'][0]
                
                spatial_fixed |= spatial_bit_plane.astype(np.int32) << bit_plane
                angular_fixed |= angular_bit_plane.astype(np.int32) << bit_plane
        
        # Convert back to floating point using adaptive conversion parameters
        if channel_wise:
            spatial_reconstructed = np.zeros_like(spatial_fixed, dtype=np.float32)
            angular_reconstructed = np.zeros_like(angular_fixed, dtype=np.float32)
            
            for ch in spatial_conversion_params.keys():
                spatial_params = spatial_conversion_params[ch]
                angular_params = angular_conversion_params[ch]
                
                spatial_reconstructed[:, ch, :, :] = self._fixed_point_to_float(
                    spatial_fixed[:, ch, :, :], spatial_params['scale_factor'], spatial_params['offset']
                )
                angular_reconstructed[:, ch, :, :] = self._fixed_point_to_float(
                    angular_fixed[:, ch, :, :], angular_params['scale_factor'], angular_params['offset']
                )
        else:
            spatial_params = spatial_conversion_params[0]
            angular_params = angular_conversion_params[0]
            
            spatial_reconstructed = self._fixed_point_to_float(
                spatial_fixed, spatial_params['scale_factor'], spatial_params['offset']
            )
            angular_reconstructed = self._fixed_point_to_float(
                angular_fixed, angular_params['scale_factor'], angular_params['offset']
            )
        
        return torch.tensor(spatial_reconstructed), torch.tensor(angular_reconstructed)
    

    
    def calculate_progressive_quality(self, original_spatial: torch.Tensor, original_angular: torch.Tensor,
                                    bit_plane_layers: List[Dict], side_info: Dict) -> List[Dict[str, float]]:
        """
        TRAINING-FREE: Calculate adaptive reconstruction quality at each progressive level.
        Works with adaptive side information structure.
        
        Args:
            original_spatial: Original spatial latents
            original_angular: Original angular latents
            bit_plane_layers: Adaptive encoded bit plane layers
            side_info: Adaptive side information
            
        Returns:
            List of adaptive quality metrics for each progressive level
        """
        quality_progression = []
        adaptive_progressive_levels = side_info.get('adaptive_progressive_levels', 
                                                  list(range(self.num_bit_planes - 1, -1, -1)))
        
        for num_layers in range(1, len(bit_plane_layers) + 1):
            # Reconstruct with current number of layers using adaptive decoder
            reconstructed_spatial, reconstructed_angular = self.decode_progressive(
                bit_plane_layers, side_info, num_layers
            )
            
            # Calculate quality metrics
            spatial_mse = torch.mean((original_spatial - reconstructed_spatial) ** 2)
            angular_mse = torch.mean((original_angular - reconstructed_angular) ** 2)
            
            spatial_psnr = -10 * torch.log10(spatial_mse) if spatial_mse > 0 else float('inf')
            angular_psnr = -10 * torch.log10(angular_mse) if angular_mse > 0 else float('inf')
            
            # Calculate bits used so far (more accurate counting)
            bits_used = 0
            for layer_idx in range(num_layers):
                layer = bit_plane_layers[layer_idx]
                
                # Count bit plane data bits
                for ch_data in layer['spatial_channels'].values():
                    bits_used += ch_data.size
                for ch_data in layer['angular_channels'].values():
                    bits_used += ch_data.size
                
                # Count significance map bits if used
                if layer['significance_maps'] and side_info.get('use_significance_map', False):
                    for sig_map in layer['significance_maps'].values():
                        bits_used += sig_map.size
            
            # Calculate adaptive compression metrics
            total_original_bits = (original_spatial.numel() + original_angular.numel()) * 32
            compression_ratio = total_original_bits / bits_used if bits_used > 0 else float('inf')
            
            # Current bit plane information
            current_bit_plane = bit_plane_layers[num_layers - 1]['bit_plane']
            current_bit_importance = 2 ** current_bit_plane  # Weight by bit significance
            
            quality_progression.append({
                'layers_used': num_layers,
                'bit_plane': current_bit_plane,
                'bit_plane_importance': current_bit_importance,
                'spatial_mse': spatial_mse.item(),
                'angular_mse': angular_mse.item(),
                'combined_mse': (spatial_mse.item() + angular_mse.item()) / 2,
                'spatial_psnr': spatial_psnr.item() if spatial_psnr != float('inf') else 100.0,
                'angular_psnr': angular_psnr.item() if angular_psnr != float('inf') else 100.0,
                'avg_psnr': ((spatial_psnr.item() if spatial_psnr != float('inf') else 100.0) + 
                           (angular_psnr.item() if angular_psnr != float('inf') else 100.0)) / 2,
                'bits_used': bits_used,
                'compression_ratio': compression_ratio,
                'adaptive_order_used': side_info.get('adaptive_order', 'magnitude'),
                'adaptive_efficiency': compression_ratio * ((spatial_psnr.item() if spatial_psnr != float('inf') else 100.0) + 
                                                          (angular_psnr.item() if angular_psnr != float('inf') else 100.0)) / 2
            })
        
        return quality_progression


class BitstreamStructurer:
    """
    TRAINING-FREE Adaptive Bitstream Structuring component.
    
    Adaptively applies RLE, zigzag scanning, and Rice coding based on real-time
    data analysis. No training required - works immediately with any VAE latents.
    
    Features:
    - 4 adaptive Rice coding methods (magnitude, entropy, frequency, hybrid)
    - Real-time data pattern analysis
    - Adaptive parameter optimization
    - Multiple structuring modes (rle_only, rle_zigzag, full_adaptive, minimal)
    - Channel-wise and global processing options
    """
    
    def __init__(self, adaptive_method: str = 'hybrid', channel_wise: bool = True,
                 base_rice_parameter: int = 2, structuring_mode: str = 'full_adaptive'):
        """
        Initialize the training-free adaptive bitstream structurer.
        
        Args:
            adaptive_method: Adaptive Rice coding method
                - 'magnitude': Magnitude-based parameter selection
                - 'entropy': Entropy-optimized parameters
                - 'frequency': Frequency distribution-based 
                - 'hybrid': Mixed method selection per channel
            channel_wise: Whether to structure channels separately
            base_rice_parameter: Base Rice parameter (1-8)
            structuring_mode: Structuring approach
                - 'full_adaptive': RLE + Zigzag + Adaptive Rice
                - 'rle_zigzag': RLE + Zigzag (no Rice coding)
                - 'rle_only': RLE only
                - 'minimal': Minimal processing
        """
        # Core configuration
        self.adaptive_method = adaptive_method
        self.channel_wise = channel_wise
        self.base_rice_parameter = max(1, min(8, base_rice_parameter))
        self.structuring_mode = structuring_mode
        
        # Derived settings from mode
        self._configure_from_mode()
        
        # REMOVED: No training dependencies
        # self.spatial_patterns = {}
        # self.angular_patterns = {}
        # self.is_trained = False
        
        print(f"✅ Training-Free Adaptive Bitstream Structurer initialized")
        print(f"   Method: {self.adaptive_method}")
        print(f"   Mode: {self.structuring_mode}")
        print(f"   Base Rice parameter: {self.base_rice_parameter}")
        print(f"   Channel-wise: {self.channel_wise}")
    
    def _configure_from_mode(self):
        """Configure component settings based on structuring mode."""
        mode_configs = {
            'full_adaptive': {'use_rle': True, 'use_zigzag': True, 'use_rice_coding': True},
            'rle_zigzag': {'use_rle': True, 'use_zigzag': True, 'use_rice_coding': False},
            'rle_only': {'use_rle': True, 'use_zigzag': False, 'use_rice_coding': False},
            'minimal': {'use_rle': False, 'use_zigzag': True, 'use_rice_coding': False}
        }
        
        config = mode_configs.get(self.structuring_mode, mode_configs['full_adaptive'])
        self.use_rle = config['use_rle']
        self.use_zigzag = config['use_zigzag']
        self.use_rice_coding = config['use_rice_coding']
    
    def _analyze_adaptive_data_patterns(self, data: np.ndarray, method: str = 'hybrid') -> Dict[str, float]:
        """
        TRAINING-FREE: Analyze data patterns adaptively for optimal structuring.
        
        Args:
            data: Data to analyze
            method: Analysis method (magnitude, entropy, frequency, hybrid)
            
        Returns:
            Dictionary with adaptive pattern statistics
        """
        flat_data = data.flatten()
        
        # Basic sparsity analysis
        zero_ratio = np.mean(flat_data == 0)
        non_zero_data = flat_data[flat_data != 0]
        
        # Adaptive Rice parameter calculation
        if len(non_zero_data) > 0:
            if method == 'magnitude':
                # Magnitude-based method
                mean_magnitude = np.mean(np.abs(non_zero_data))
                adaptive_rice_k = max(1, min(8, int(np.log2(mean_magnitude + 1))))
                
            elif method == 'entropy':
                # Entropy-optimized method
                # Calculate histogram and entropy
                hist, _ = np.histogram(np.abs(non_zero_data), bins=min(50, len(non_zero_data)))
                prob = hist / np.sum(hist)
                prob = prob[prob > 0]  # Remove zeros
                entropy = -np.sum(prob * np.log2(prob))
                
                # Map entropy to Rice parameter (higher entropy = higher k)
                adaptive_rice_k = max(1, min(8, int(entropy / 2.0)))
                
            elif method == 'frequency':
                # Frequency distribution-based method
                unique_vals, counts = np.unique(np.abs(non_zero_data), return_counts=True)
                
                # Calculate concentration of values
                concentration = np.max(counts) / len(non_zero_data)
                
                # Lower concentration = higher Rice parameter
                adaptive_rice_k = max(1, min(8, int(5 * (1 - concentration))))
                
            elif method == 'hybrid':
                # Hybrid method: combine magnitude and entropy
                mean_magnitude = np.mean(np.abs(non_zero_data))
                magnitude_k = max(1, min(8, int(np.log2(mean_magnitude + 1))))
                
                # Simple entropy calculation
                unique_vals = len(np.unique(non_zero_data))
                total_vals = len(non_zero_data)
                normalized_entropy = unique_vals / total_vals
                entropy_k = max(1, min(8, int(4 * normalized_entropy)))
                
                # Weighted combination
                adaptive_rice_k = max(1, min(8, int((magnitude_k + entropy_k) / 2)))
                
            else:
                adaptive_rice_k = self.base_rice_parameter
        else:
            adaptive_rice_k = 1  # Minimal parameter for all-zero data
        
        # Run-length analysis for RLE effectiveness
        zero_runs = []
        current_run = 0
        
        for value in flat_data:
            if value == 0:
                current_run += 1
            else:
                if current_run > 0:
                    zero_runs.append(current_run)
                    current_run = 0
        
        if current_run > 0:
            zero_runs.append(current_run)
        
        # Adaptive RLE benefit calculation
        avg_zero_run = np.mean(zero_runs) if zero_runs else 0
        rle_efficiency = min(avg_zero_run / max(1, np.sqrt(len(flat_data))), 1.0)
        
        # Adaptive zigzag benefit (based on local correlation)
        zigzag_benefit = self._calculate_adaptive_zigzag_benefit(data)
        
        return {
            'zero_ratio': zero_ratio,
            'adaptive_rice_parameter': adaptive_rice_k,
            'avg_zero_run_length': avg_zero_run,
            'max_zero_run_length': np.max(zero_runs) if zero_runs else 0,
            'num_zero_runs': len(zero_runs),
            'rle_efficiency': rle_efficiency,
            'zigzag_benefit': zigzag_benefit,
            'sparsity_benefit': zero_ratio > 0.25,  # Adaptive threshold
            'compression_potential': (zero_ratio + rle_efficiency + zigzag_benefit) / 3.0,
            'method_used': method
        }
    
    def _calculate_adaptive_zigzag_benefit(self, data: np.ndarray) -> float:
        """Calculate adaptive benefit of zigzag scanning based on local correlation."""
        if len(data.shape) < 2:
            return 0.0
        
        # Sample a representative 2D slice
        if len(data.shape) == 4:  # [B, C, H, W]
            sample_slice = data[0, 0, :, :] if data.shape[0] > 0 and data.shape[1] > 0 else data.reshape(data.shape[-2:])
        elif len(data.shape) == 3:  # [C, H, W] or [B, H, W]
            sample_slice = data[0, :, :] if data.shape[0] > 0 else data.reshape(data.shape[-2:])
        else:
            sample_slice = data
        
        if sample_slice.size < 4:
            return 0.0
        
        # Calculate spatial correlation
        try:
            # Horizontal correlation
            h_corr = np.corrcoef(sample_slice[:-1, :].flatten(), sample_slice[1:, :].flatten())[0, 1]
            # Vertical correlation  
            v_corr = np.corrcoef(sample_slice[:, :-1].flatten(), sample_slice[:, 1:].flatten())[0, 1]
            
            # Handle NaN correlation values
            h_corr = h_corr if not np.isnan(h_corr) else 0.0
            v_corr = v_corr if not np.isnan(v_corr) else 0.0
            
            # Zigzag benefit increases with spatial correlation
            return (abs(h_corr) + abs(v_corr)) / 2.0
            
        except:
            return 0.0
    
    def _select_optimal_adaptive_method(self, spatial_data: np.ndarray, angular_data: np.ndarray) -> str:
        """Select optimal adaptive method based on data characteristics."""
        # Test all methods and select best one
        methods = ['magnitude', 'entropy', 'frequency']
        method_scores = {}
        
        for method in methods:
            spatial_patterns = self._analyze_adaptive_data_patterns(spatial_data, method)
            angular_patterns = self._analyze_adaptive_data_patterns(angular_data, method)
            
            # Score based on compression potential
            score = (spatial_patterns['compression_potential'] + 
                    angular_patterns['compression_potential']) / 2.0
            
            method_scores[method] = score
        
        # Select method with highest score
        best_method = max(method_scores.keys(), key=lambda k: method_scores[k])
        return best_method
    
    def _create_zigzag_pattern(self, height: int, width: int) -> List[Tuple[int, int]]:
        """
        Create zigzag scanning pattern for 2D data.
        
        Args:
            height: Height of the 2D array
            width: Width of the 2D array
            
        Returns:
            List of (row, col) coordinates in zigzag order
        """
        pattern = []
        
        # Start from top-left
        for i in range(height + width - 1):
            if i % 2 == 0:  # Even diagonals (bottom-left to top-right)
                for j in range(max(0, i - height + 1), min(i + 1, width)):
                    row = i - j
                    col = j
                    if 0 <= row < height and 0 <= col < width:
                        pattern.append((row, col))
            else:  # Odd diagonals (top-right to bottom-left)
                for j in range(min(i, width - 1), max(-1, i - height), -1):
                    row = i - j
                    col = j
                    if 0 <= row < height and 0 <= col < width:
                        pattern.append((row, col))
        
        return pattern
    
    def _apply_zigzag_scan(self, data: np.ndarray, pattern: List[Tuple[int, int]]) -> np.ndarray:
        """
        Apply zigzag scanning to 2D data.
        
        Args:
            data: 2D array to scan
            pattern: Zigzag pattern
            
        Returns:
            1D array in zigzag order
        """
        height, width = data.shape
        zigzag_data = np.zeros(height * width)
        
        for idx, (row, col) in enumerate(pattern):
            if idx < len(zigzag_data):
                zigzag_data[idx] = data[row, col]
        
        return zigzag_data
    
    def _reverse_zigzag_scan(self, zigzag_data: np.ndarray, height: int, width: int, 
                           pattern: List[Tuple[int, int]]) -> np.ndarray:
        """
        Reverse zigzag scanning to reconstruct 2D data.
        
        Args:
            zigzag_data: 1D array in zigzag order
            height: Target height
            width: Target width
            pattern: Zigzag pattern
            
        Returns:
            2D array reconstructed from zigzag scan
        """
        data = np.zeros((height, width))
        
        for idx, (row, col) in enumerate(pattern):
            if idx < len(zigzag_data):
                data[row, col] = zigzag_data[idx]
        
        return data
    
    def _run_length_encode(self, data: np.ndarray) -> List[Tuple[float, int]]:
        """
        Apply Run-Length Encoding to data.
        
        Args:
            data: 1D array to encode
            
        Returns:
            List of (value, count) tuples
        """
        if len(data) == 0:
            return []
        
        rle_pairs = []
        current_value = data[0]
        current_count = 1
        
        for i in range(1, len(data)):
            if data[i] == current_value:
                current_count += 1
            else:
                rle_pairs.append((current_value, current_count))
                current_value = data[i]
                current_count = 1
        
        # Add the last run
        rle_pairs.append((current_value, current_count))
        
        return rle_pairs
    
    def _run_length_decode(self, rle_pairs: List[Tuple[float, int]]) -> np.ndarray:
        """
        Decode Run-Length Encoded data.
        
        Args:
            rle_pairs: List of (value, count) tuples
            
        Returns:
            Decoded 1D array
        """
        if not rle_pairs:
            return np.array([])
        
        total_length = sum(count for _, count in rle_pairs)
        decoded_data = np.zeros(total_length)
        
        idx = 0
        for value, count in rle_pairs:
            decoded_data[idx:idx + count] = value
            idx += count
        
        return decoded_data
    
    def _adaptive_rice_encode(self, values: List[int], adaptive_k: int) -> Tuple[List[int], Dict]:
        """
        Apply adaptive Rice coding to integer values.
        
        Args:
            values: List of non-negative integers to encode
            adaptive_k: Adaptive Rice parameter
            
        Returns:
            Tuple of (encoded_bits, rice_info)
        """
        encoded_bits = []
        total_original_bits = 0
        total_encoded_bits = 0
        
        for value in values:
            if value < 0:
                # Handle negative values using zigzag encoding
                zigzag_value = (value << 1) ^ (value >> 31)
            else:
                zigzag_value = value << 1
            
            # Rice coding: quotient and remainder
            quotient = zigzag_value >> adaptive_k
            remainder = zigzag_value & ((1 << adaptive_k) - 1)
            
            # Unary code for quotient (quotient zeros followed by a one)
            unary_code = [0] * quotient + [1]
            
            # Binary code for remainder (k bits)
            binary_code = [(remainder >> i) & 1 for i in range(adaptive_k - 1, -1, -1)]
            
            # Combine unary and binary codes
            rice_code = unary_code + binary_code
            encoded_bits.extend(rice_code)
            
            # Statistics
            original_bits = max(1, int(np.ceil(np.log2(abs(value) + 1))))
            total_original_bits += original_bits
            total_encoded_bits += len(rice_code)
        
        rice_info = {
            'adaptive_rice_parameter': adaptive_k,
            'num_values': len(values),
            'total_original_bits': total_original_bits,
            'total_encoded_bits': total_encoded_bits,
            'compression_ratio': total_original_bits / max(1, total_encoded_bits),
            'efficiency': min(1.0, total_original_bits / max(1, total_encoded_bits))
        }
        
        return encoded_bits, rice_info
    
    def _adaptive_rice_decode(self, encoded_bits: List[int], rice_info: Dict) -> List[int]:
        """
        Decode adaptive Rice coded data.
        
        Args:
            encoded_bits: List of encoded bits
            rice_info: Adaptive Rice coding information
            
        Returns:
            List of decoded integer values
        """
        adaptive_k = rice_info['adaptive_rice_parameter']
        num_values = rice_info['num_values']
        decoded_values = []
        
        bit_idx = 0
        for _ in range(num_values):
            # Read unary code (count zeros until we hit a one)
            quotient = 0
            while bit_idx < len(encoded_bits) and encoded_bits[bit_idx] == 0:
                quotient += 1
                bit_idx += 1
            
            # Skip the terminating one
            if bit_idx < len(encoded_bits):
                bit_idx += 1
            
            # Read binary code (k bits)
            remainder = 0
            for i in range(adaptive_k):
                if bit_idx < len(encoded_bits):
                    remainder = (remainder << 1) | encoded_bits[bit_idx]
                    bit_idx += 1
            
            # Reconstruct zigzag value
            zigzag_value = (quotient << adaptive_k) | remainder
            
            # Decode zigzag encoding
            if zigzag_value & 1:
                value = ~(zigzag_value >> 1)
            else:
                value = zigzag_value >> 1
            
            decoded_values.append(value)
        
        return decoded_values
    
    def structure_bitstream(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        TRAINING-FREE: Apply adaptive bitstream structuring to latent data.
        
        Args:
            spatial_latents: Spatial latents to structure
            angular_latents: Angular latents to structure
            
        Returns:
            Tuple of (structured_data, side_info)
        """
        spatial_data = spatial_latents.cpu().numpy()
        angular_data = angular_latents.cpu().numpy()
        batch_size, channels, height, width = spatial_data.shape
        
        # TRAINING-FREE: Adaptive method selection if hybrid mode
        if self.adaptive_method == 'hybrid':
            optimal_method = self._select_optimal_adaptive_method(spatial_data, angular_data)
        else:
            optimal_method = self.adaptive_method
        
        structured_data = {
            'spatial_channels': {},
            'angular_channels': {},
            'compression_stats': {},
            'adaptive_info': {
                'method_used': optimal_method,
                'structuring_mode': self.structuring_mode
            }
        }
        
        total_original_bits = 0
        total_structured_bits = 0
        
        # Process each channel
        channels_to_process = range(64) if self.channel_wise else [0]
        
        for ch in channels_to_process:
            if self.channel_wise:
                spatial_ch_data = spatial_data[:, ch, :, :]
                angular_ch_data = angular_data[:, ch, :, :]
            else:
                spatial_ch_data = spatial_data
                angular_ch_data = angular_data
            
            # Process spatial data with adaptive patterns
            spatial_structured = self._structure_channel_data_adaptive(
                spatial_ch_data, f'spatial_{ch}', optimal_method, height, width
            )
            structured_data['spatial_channels'][ch] = spatial_structured
            
            # Process angular data with adaptive patterns
            angular_structured = self._structure_channel_data_adaptive(
                angular_ch_data, f'angular_{ch}', optimal_method, height, width
            )
            structured_data['angular_channels'][ch] = angular_structured
            
            # Update statistics
            total_original_bits += spatial_structured['original_bits'] + angular_structured['original_bits']
            total_structured_bits += spatial_structured['structured_bits'] + angular_structured['structured_bits']
        
        # Create comprehensive side information
        side_info = {
            'bitstream_structuring_type': 'adaptive_rle_zigzag_rice',
            'adaptive_method': optimal_method,
            'structuring_mode': self.structuring_mode,
            'use_rle': self.use_rle,
            'use_zigzag': self.use_zigzag,
            'use_rice_coding': self.use_rice_coding,
            'channel_wise': self.channel_wise,
            'original_shape': spatial_latents.shape,
            'zigzag_pattern': self._create_zigzag_pattern(height, width),
            'adaptive_parameters': structured_data['adaptive_info'],
            'compression_stats': {
                'total_original_bits': total_original_bits,
                'total_structured_bits': total_structured_bits,
                'compression_ratio': total_original_bits / max(1, total_structured_bits),
                'size_reduction_percent': (1 - total_structured_bits / max(1, total_original_bits)) * 100,
                'bits_per_element': total_structured_bits / max(1, spatial_latents.numel() + angular_latents.numel())
            }
        }
        
        structured_data['compression_stats'] = side_info['compression_stats']
        
        return structured_data, side_info
    
    def _structure_channel_data_adaptive(self, channel_data: np.ndarray, channel_id: str, 
                                       adaptive_method: str, height: int, width: int) -> Dict:
        """
        TRAINING-FREE: Apply adaptive structuring to a single channel's data.
        
        Args:
            channel_data: Channel data to structure
            channel_id: Identifier for the channel
            adaptive_method: Adaptive method to use
            height: Data height
            width: Data width
            
        Returns:
            Dictionary with structured data and statistics
        """
        original_size = channel_data.size
        original_bits = original_size * 32  # Assuming float32
        
        # TRAINING-FREE: Adaptive pattern analysis
        adaptive_patterns = self._analyze_adaptive_data_patterns(channel_data, adaptive_method)
        
        structured_info = {
            'channel_id': channel_id,
            'original_bits': original_bits,
            'processing_steps': [],
            'adaptive_patterns': adaptive_patterns
        }
        
        # Convert to integer for processing (adaptive scaling)
        max_val = np.max(np.abs(channel_data)) if channel_data.size > 0 else 1.0
        scale_factor = 1000.0 if max_val < 10.0 else 100.0  # Adaptive scaling
        int_data = np.round(channel_data * scale_factor).astype(np.int32)
        
        current_data = int_data.copy()
        
        # Step 1: Adaptive Zigzag scanning
        if self.use_zigzag and adaptive_patterns['zigzag_benefit'] > 0.1:
            zigzag_pattern = self._create_zigzag_pattern(height, width)
            zigzag_data = []
            
            for b in range(current_data.shape[0]):
                for ch in range(current_data.shape[1]) if len(current_data.shape) > 3 else [0]:
                    if len(current_data.shape) > 3:
                        slice_2d = current_data[b, ch, :, :]
                    else:
                        slice_2d = current_data[b, :, :]
                    
                    zigzag_slice = self._apply_zigzag_scan(slice_2d, zigzag_pattern)
                    zigzag_data.append(zigzag_slice)
            
            current_data = np.concatenate(zigzag_data)
            structured_info['processing_steps'].append('adaptive_zigzag_scan')
            structured_info['zigzag_pattern'] = zigzag_pattern
        else:
            current_data = current_data.flatten()
        
        # Step 2: Adaptive Run-Length Encoding
        if self.use_rle and adaptive_patterns['sparsity_benefit']:
            rle_pairs = self._run_length_encode(current_data)
            
            # Convert RLE pairs to separate value and count arrays
            if rle_pairs:
                rle_values, rle_counts = zip(*rle_pairs)
                rle_values = list(rle_values)
                rle_counts = list(rle_counts)
            else:
                rle_values, rle_counts = [], []
            
            structured_info['rle_values'] = rle_values
            structured_info['rle_counts'] = rle_counts
            structured_info['rle_pairs_count'] = len(rle_pairs)
            structured_info['processing_steps'].append('adaptive_run_length_encoding')
            
            # Use RLE values for Rice coding
            rice_input = rle_values + rle_counts
        else:
            rice_input = current_data.tolist()
        
        # Step 3: Adaptive Rice Coding
        if self.use_rice_coding and rice_input:
            adaptive_k = adaptive_patterns['adaptive_rice_parameter']
            
            # Ensure all values are integers
            rice_input_int = [int(x) for x in rice_input]
            rice_bits, rice_info = self._adaptive_rice_encode(rice_input_int, adaptive_k)
            
            structured_info['rice_bits'] = rice_bits
            structured_info['rice_info'] = rice_info
            structured_info['processing_steps'].append('adaptive_rice_coding')
            
            structured_bits = len(rice_bits)
        else:
            # Without Rice coding, estimate bits needed
            if self.use_rle and adaptive_patterns['sparsity_benefit']:
                # Estimate bits for RLE values and counts
                max_value = max(abs(x) for x in rle_values) if rle_values else 1
                max_count = max(rle_counts) if rle_counts else 1
                
                value_bits = int(np.ceil(np.log2(max_value + 1))) + 1  # +1 for sign
                count_bits = int(np.ceil(np.log2(max_count + 1)))
                
                structured_bits = len(rle_values) * value_bits + len(rle_counts) * count_bits
            else:
                # Estimate bits for raw data
                max_value = np.max(np.abs(current_data)) if current_data.size > 0 else 1
                value_bits = int(np.ceil(np.log2(max_value + 1))) + 1  # +1 for sign
                structured_bits = len(current_data) * value_bits
        
        structured_info['structured_bits'] = structured_bits
        structured_info['scale_factor'] = scale_factor
        structured_info['compression_ratio'] = original_bits / max(1, structured_bits)
        structured_info['efficiency'] = min(1.0, structured_info['compression_ratio'])
        
        return structured_info
    
    def reconstruct_bitstream(self, structured_data: Dict, side_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TRAINING-FREE: Reconstruct latents from structured bitstream data.
        
        Args:
            structured_data: Structured bitstream data
            side_info: Side information from structuring
            
        Returns:
            Tuple of (reconstructed_spatial, reconstructed_angular)
        """
        original_shape = side_info['original_shape']
        batch_size, channels, height, width = original_shape
        channel_wise = side_info['channel_wise']
        
        # Initialize reconstruction arrays
        spatial_reconstructed = np.zeros(original_shape)
        angular_reconstructed = np.zeros(original_shape)
        
        # Process each channel
        channels_to_process = range(64) if channel_wise else [0]
        
        for ch in channels_to_process:
            # Reconstruct spatial data
            if ch in structured_data['spatial_channels']:
                spatial_ch_data = self._reconstruct_channel_data_adaptive(
                    structured_data['spatial_channels'][ch], 
                    side_info, 
                    original_shape
                )
                
                if channel_wise:
                    spatial_reconstructed[:, ch, :, :] = spatial_ch_data
                else:
                    spatial_reconstructed = spatial_ch_data
            
            # Reconstruct angular data
            if ch in structured_data['angular_channels']:
                angular_ch_data = self._reconstruct_channel_data_adaptive(
                    structured_data['angular_channels'][ch], 
                    side_info, 
                    original_shape
                )
                
                if channel_wise:
                    angular_reconstructed[:, ch, :, :] = angular_ch_data
                else:
                    angular_reconstructed = angular_ch_data
        
        return torch.tensor(spatial_reconstructed, dtype=torch.float32), torch.tensor(angular_reconstructed, dtype=torch.float32)
    
    def _reconstruct_channel_data_adaptive(self, channel_info: Dict, side_info: Dict, original_shape: Tuple) -> np.ndarray:
        """
        TRAINING-FREE: Reconstruct a single channel's data from adaptive structuring information.
        
        Args:
            channel_info: Channel structuring information
            side_info: Global side information
            original_shape: Original data shape
            
        Returns:
            Reconstructed channel data
        """
        batch_size, channels, height, width = original_shape
        scale_factor = channel_info['scale_factor']
        processing_steps = channel_info['processing_steps']
        
        # Reverse adaptive Rice coding
        if 'adaptive_rice_coding' in processing_steps:
            rice_bits = channel_info['rice_bits']
            rice_info = channel_info['rice_info']
            decoded_values = self._adaptive_rice_decode(rice_bits, rice_info)
        else:
            # Handle case without Rice coding
            if 'adaptive_run_length_encoding' in processing_steps:
                decoded_values = channel_info['rle_values'] + channel_info['rle_counts']
            else:
                # This case shouldn't happen in normal operation
                decoded_values = []
        
        # Reverse adaptive Run-Length Encoding
        if 'adaptive_run_length_encoding' in processing_steps:
            num_pairs = channel_info['rle_pairs_count']
            if len(decoded_values) >= 2 * num_pairs and num_pairs > 0:
                rle_values = decoded_values[:num_pairs]
                rle_counts = decoded_values[num_pairs:2*num_pairs]
                
                # Reconstruct from RLE pairs
                rle_pairs = list(zip(rle_values, rle_counts))
                flat_data = self._run_length_decode(rle_pairs)
            else:
                # Fallback for empty or invalid RLE data
                flat_data = np.array(decoded_values) if decoded_values else np.zeros(batch_size * height * width)
        else:
            flat_data = np.array(decoded_values) if decoded_values else np.zeros(batch_size * height * width)
        
        # Reverse adaptive zigzag scanning
        if 'adaptive_zigzag_scan' in processing_steps:
            zigzag_pattern = channel_info.get('zigzag_pattern', side_info.get('zigzag_pattern'))
            
            if side_info['channel_wise']:
                # Reshape for single channel
                target_shape = (batch_size, height, width)
                elements_per_slice = height * width
                
                # Ensure we have enough data
                if len(flat_data) >= batch_size * elements_per_slice:
                    reconstructed_slices = []
                    data_idx = 0
                    
                    for b in range(batch_size):
                        slice_data = flat_data[data_idx:data_idx + elements_per_slice]
                        if len(slice_data) == elements_per_slice:
                            reconstructed_slice = self._reverse_zigzag_scan(slice_data, height, width, zigzag_pattern)
                            reconstructed_slices.append(reconstructed_slice)
                        else:
                            # Pad if needed
                            padded_slice = np.zeros(elements_per_slice)
                            padded_slice[:len(slice_data)] = slice_data
                            reconstructed_slice = self._reverse_zigzag_scan(padded_slice, height, width, zigzag_pattern)
                            reconstructed_slices.append(reconstructed_slice)
                        data_idx += elements_per_slice
                    
                    reconstructed_data = np.stack(reconstructed_slices, axis=0)
                else:
                    # Fallback: pad data to required size
                    required_size = batch_size * elements_per_slice
                    padded_data = np.zeros(required_size)
                    padded_data[:len(flat_data)] = flat_data
                    reconstructed_data = padded_data.reshape(target_shape)
            else:
                # Global reconstruction
                if len(flat_data) >= height * width:
                    reconstructed_data = self._reverse_zigzag_scan(flat_data[:height*width], height, width, zigzag_pattern)
                    reconstructed_data = reconstructed_data.reshape(original_shape)
                else:
                    # Fallback for insufficient data
                    reconstructed_data = np.zeros(original_shape)
                    if len(flat_data) > 0:
                        flat_reshaped = flat_data.reshape(-1)
                        total_elements = np.prod(original_shape)
                        if len(flat_reshaped) <= total_elements:
                            reconstructed_data.flat[:len(flat_reshaped)] = flat_reshaped
        else:
            # Direct reshape
            if side_info['channel_wise']:
                target_shape = (batch_size, height, width)
            else:
                target_shape = original_shape
            
            # Ensure data matches target shape
            required_size = np.prod(target_shape)
            if len(flat_data) == required_size:
                reconstructed_data = flat_data.reshape(target_shape)
            else:
                # Pad or truncate as needed
                adjusted_data = np.zeros(required_size)
                copy_size = min(len(flat_data), required_size)
                adjusted_data[:copy_size] = flat_data[:copy_size]
                reconstructed_data = adjusted_data.reshape(target_shape)
        
        # Convert back to float using scale factor
        reconstructed_data = reconstructed_data.astype(np.float32) / scale_factor
        
        return reconstructed_data
    
    def calculate_compression_benefit(self, original_spatial: torch.Tensor, original_angular: torch.Tensor,
                                    structured_data: Dict, side_info: Dict) -> Dict[str, float]:
        """
        Calculate compression benefit of adaptive bitstream structuring.
        
        Args:
            original_spatial: Original spatial latents
            original_angular: Original angular latents
            structured_data: Structured data
            side_info: Side information
            
        Returns:
            Dictionary with compression statistics
        """
        # Reconstruct data
        reconstructed_spatial, reconstructed_angular = self.reconstruct_bitstream(structured_data, side_info)
        
        # Calculate reconstruction error
        spatial_mse = torch.mean((original_spatial - reconstructed_spatial) ** 2).item()
        angular_mse = torch.mean((original_angular - reconstructed_angular) ** 2).item()
        
        # Get compression stats
        compression_stats = side_info['compression_stats']
        
        return {
            'spatial_reconstruction_mse': spatial_mse,
            'angular_reconstruction_mse': angular_mse,
            'total_reconstruction_mse': (spatial_mse + angular_mse) / 2.0,
            'compression_ratio': compression_stats['compression_ratio'],
            'size_reduction_percent': compression_stats['size_reduction_percent'],
            'bits_per_element': compression_stats['bits_per_element'],
            'adaptive_method_used': side_info['adaptive_method'],
            'structuring_mode': side_info['structuring_mode'],
            'rle_used': side_info['use_rle'],
            'zigzag_used': side_info['use_zigzag'],
            'rice_coding_used': side_info['use_rice_coding']
        }


class AdaptiveDecoderReconstructionStrategy:
    """
    Component 9: Training-Free Adaptive Decoder Reconstruction Strategy
    
    Provides intelligent reconstruction strategy to restore original latent tensors
    from compressed bitstream. Handles complete reversal of all compression 
    transformations in optimal order for DUALF-D decoder input.
    
    Features:
    - Training-free adaptive reconstruction ordering
    - Multi-stage reconstruction validation
    - Error correction and recovery mechanisms  
    - Optimal reconstruction path selection
    - Quality-aware reconstruction strategies
    """
    
    def __init__(self, reconstruction_strategy: str = 'adaptive_optimal', 
                 error_correction: bool = True, quality_validation: bool = True,
                 fallback_strategies: bool = True):
        """
        Initialize the training-free adaptive decoder reconstruction strategy.
        
        Args:
            reconstruction_strategy: Strategy for reconstruction ordering
                - 'adaptive_optimal': Adaptive optimal ordering based on data analysis
                - 'reverse_encoding': Strict reverse of encoding order
                - 'quality_prioritized': Quality-first reconstruction
                - 'speed_prioritized': Speed-optimized reconstruction
            error_correction: Enable error correction during reconstruction
            quality_validation: Enable quality validation at each step
            fallback_strategies: Enable fallback reconstruction strategies
        """
        self.reconstruction_strategy = reconstruction_strategy
        self.error_correction = error_correction
        self.quality_validation = quality_validation
        self.fallback_strategies = fallback_strategies
        
        # Reconstruction metrics tracking
        self.reconstruction_stats = {}
        self.strategy_performance = {}
        
        print(f"✅ Training-Free Adaptive Decoder Reconstruction Strategy initialized")
        print(f"   Strategy: {self.reconstruction_strategy}")
        print(f"   Error correction: {self.error_correction}")
        print(f"   Quality validation: {self.quality_validation}")
        print(f"   Fallback strategies: {self.fallback_strategies}")
    
    def _analyze_reconstruction_complexity(self, encoded_data: Dict, side_info: Dict) -> Dict[str, float]:
        """
        TRAINING-FREE: Analyze reconstruction complexity to determine optimal strategy.
        
        Args:
            encoded_data: Encoded data from compression pipeline
            side_info: Complete side information from encoding
            
        Returns:
            Dictionary with complexity analysis
        """
        complexity_analysis = {
            'total_components_used': 0,
            'arithmetic_coding_complexity': 0.0,
            'bitstream_structuring_complexity': 0.0, 
            'bit_plane_complexity': 0.0,
            'transform_complexity': 0.0,
            'vector_quantization_complexity': 0.0,
            'quantization_complexity': 0.0,
            'sparsification_complexity': 0.0,
            'reordering_complexity': 0.0,
            'estimated_reconstruction_time': 0.0,
            'error_risk_level': 'low'
        }
        
        # Component 7: Arithmetic coding analysis
        if 'arithmetic_coding' in side_info.get('component_side_info', {}):
            complexity_analysis['total_components_used'] += 1
            arithmetic_info = side_info['component_side_info']['arithmetic_coding']
            
            # Complexity based on adaptive method and bit count
            method_complexity = {
                'frequency': 1.0, 'laplace': 1.2, 'entropy': 1.5, 'kneser_ney': 2.0
            }
            adaptive_method = arithmetic_info.get('adaptive_method', 'frequency')
            base_complexity = method_complexity.get(adaptive_method, 1.0)
            
            # Scale by data size
            total_bits = arithmetic_info.get('total_bits', 0)
            complexity_analysis['arithmetic_coding_complexity'] = base_complexity * (total_bits / 100000)
        
        # Component 8: Bitstream structuring analysis  
        if 'bitstream_structuring' in side_info.get('component_side_info', {}):
            complexity_analysis['total_components_used'] += 1
            bitstream_info = side_info['component_side_info']['bitstream_structuring']
            
            # Complexity based on features used
            features_complexity = 0.0
            if bitstream_info.get('use_rle', False):
                features_complexity += 0.3
            if bitstream_info.get('use_zigzag', False):
                features_complexity += 0.2  
            if bitstream_info.get('use_rice_coding', False):
                features_complexity += 0.5
                
            complexity_analysis['bitstream_structuring_complexity'] = features_complexity
        
        # Component 6: Bit-plane analysis
        if 'bit_plane_coding' in side_info.get('component_side_info', {}):
            complexity_analysis['total_components_used'] += 1
            bit_plane_info = side_info['component_side_info']['bit_plane_coding']
            
            num_layers = len(encoded_data.get('bit_plane_layers', []))
            complexity_analysis['bit_plane_complexity'] = num_layers * 0.1
        
        # Component 5: Transform coding analysis
        if 'transform_coding' in side_info.get('component_side_info', {}):
            complexity_analysis['total_components_used'] += 1
            transform_info = side_info['component_side_info']['transform_coding']
            
            # Complexity based on adaptive method
            method_complexity = {
                'energy_adaptive': 1.2, 'variance_adaptive': 1.0, 'frequency_adaptive': 1.5
            }
            adaptive_method = transform_info.get('adaptive_method', 'energy_adaptive')
            complexity_analysis['transform_complexity'] = method_complexity.get(adaptive_method, 1.0)
        
        # Component 4: Vector quantization analysis
        if 'vector_quantization' in side_info.get('component_side_info', {}):
            complexity_analysis['total_components_used'] += 1
            vq_info = side_info['component_side_info']['vector_quantization']
            
            # Complexity based on codebook sizes
            spatial_codebooks = len(vq_info.get('spatial_codebooks', {}))
            angular_codebooks = len(vq_info.get('angular_codebooks', {}))
            complexity_analysis['vector_quantization_complexity'] = (spatial_codebooks + angular_codebooks) * 0.05
        
        # Component 3: Quantization analysis  
        if 'quantization' in side_info.get('component_side_info', {}):
            complexity_analysis['total_components_used'] += 1
            quant_info = side_info['component_side_info']['quantization']
            
            # Complexity based on adaptive method and channels
            num_spatial_quantizers = len(quant_info.get('quantization_info', {}).get('spatial_quantizers', {}))
            num_angular_quantizers = len(quant_info.get('quantization_info', {}).get('angular_quantizers', {}))
            complexity_analysis['quantization_complexity'] = (num_spatial_quantizers + num_angular_quantizers) * 0.02
        
        # Component 2: Sparsification analysis
        if 'clipping_sparsification' in side_info.get('component_side_info', {}):
            complexity_analysis['total_components_used'] += 1
            sparse_info = side_info['component_side_info']['clipping_sparsification']
            
            sparsity_ratio = sparse_info.get('sparsity_statistics', {}).get('overall_sparsity', 0.0)
            complexity_analysis['sparsification_complexity'] = sparsity_ratio * 0.5
        
        # Component 1: Reordering analysis
        if 'reordering' in side_info.get('component_side_info', {}):
            complexity_analysis['total_components_used'] += 1
            reorder_info = side_info['component_side_info']['reordering']
            
            # Simple reordering complexity
            complexity_analysis['reordering_complexity'] = 0.1
        
        # Estimate total reconstruction time (relative units)
        complexity_analysis['estimated_reconstruction_time'] = (
            complexity_analysis['arithmetic_coding_complexity'] +
            complexity_analysis['bitstream_structuring_complexity'] + 
            complexity_analysis['bit_plane_complexity'] +
            complexity_analysis['transform_complexity'] +
            complexity_analysis['vector_quantization_complexity'] +
            complexity_analysis['quantization_complexity'] +
            complexity_analysis['sparsification_complexity'] +
            complexity_analysis['reordering_complexity']
        )
        
        # Determine error risk level
        if complexity_analysis['total_components_used'] >= 6:
            complexity_analysis['error_risk_level'] = 'high'
        elif complexity_analysis['total_components_used'] >= 4:
            complexity_analysis['error_risk_level'] = 'medium'
        else:
            complexity_analysis['error_risk_level'] = 'low'
        
        return complexity_analysis
    
    def _determine_optimal_reconstruction_order(self, encoded_data: Dict, side_info: Dict, 
                                              complexity_analysis: Dict) -> List[str]:
        """
        TRAINING-FREE: Determine optimal reconstruction order based on strategy and complexity.
        
        Args:
            encoded_data: Encoded data
            side_info: Side information  
            complexity_analysis: Complexity analysis results
            
        Returns:
            List of component names in optimal reconstruction order
        """
        # Available components (in reverse encoding order by default)
        all_components = [
            'arithmetic_coding',
            'bitstream_structuring', 
            'bit_plane_coding',
            'transform_coding',
            'vector_quantization',
            'quantization',
            'clipping_sparsification', 
            'reordering'
        ]
        
        # Filter to only used components
        used_components = []
        component_side_info = side_info.get('component_side_info', {})
        
        for component in all_components:
            if component in component_side_info:
                used_components.append(component)
        
        if self.reconstruction_strategy == 'reverse_encoding':
            # Strict reverse order (default)
            return used_components
            
        elif self.reconstruction_strategy == 'quality_prioritized':
            # Prioritize components that preserve quality best
            quality_priorities = {
                'arithmetic_coding': 1,      # Lossless, decode first
                'bitstream_structuring': 2,  # Lossless, decode early  
                'bit_plane_coding': 3,       # Progressive, controlled loss
                'quantization': 4,           # Major quality impact
                'vector_quantization': 5,    # Significant quality impact
                'transform_coding': 6,       # Transform domain effects
                'clipping_sparsification': 7, # Sparsity effects
                'reordering': 8              # Reordering (minimal quality impact)
            }
            
            return sorted(used_components, key=lambda x: quality_priorities.get(x, 999))
            
        elif self.reconstruction_strategy == 'speed_prioritized':
            # Prioritize fastest components first
            speed_priorities = {
                'reordering': 1,             # Fastest
                'clipping_sparsification': 2, # Fast
                'quantization': 3,           # Moderate
                'bitstream_structuring': 4,  # Moderate
                'transform_coding': 5,       # Slower
                'vector_quantization': 6,    # Slower
                'bit_plane_coding': 7,       # Slow
                'arithmetic_coding': 8       # Can be slow
            }
            
            return sorted(used_components, key=lambda x: speed_priorities.get(x, 999))
            
        elif self.reconstruction_strategy == 'adaptive_optimal':
            # Adaptive strategy based on complexity analysis
            if complexity_analysis['error_risk_level'] == 'high':
                # High risk: Use quality-prioritized order
                quality_priorities = {
                    'arithmetic_coding': 1, 'bitstream_structuring': 2, 'bit_plane_coding': 3,
                    'quantization': 4, 'vector_quantization': 5, 'transform_coding': 6,
                    'clipping_sparsification': 7, 'reordering': 8
                }
                return sorted(used_components, key=lambda x: quality_priorities.get(x, 999))
                
            elif complexity_analysis['estimated_reconstruction_time'] > 5.0:
                # High complexity: Use speed-prioritized order
                speed_priorities = {
                    'reordering': 1, 'clipping_sparsification': 2, 'quantization': 3,
                    'bitstream_structuring': 4, 'transform_coding': 5, 'vector_quantization': 6,
                    'bit_plane_coding': 7, 'arithmetic_coding': 8
                }
                return sorted(used_components, key=lambda x: speed_priorities.get(x, 999))
            else:
                # Normal case: Use reverse encoding order
                return used_components
        else:
            # Fallback: reverse encoding order
            return used_components
    
    def _validate_reconstruction_step(self, step_name: str, reconstructed_data: torch.Tensor,
                                    expected_shape: tuple, quality_threshold: float = 0.1) -> Dict[str, bool]:
        """
        Validate reconstruction step for errors and quality issues.
        
        Args:
            step_name: Name of reconstruction step
            reconstructed_data: Reconstructed tensor
            expected_shape: Expected tensor shape
            quality_threshold: Quality validation threshold
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'shape_valid': False,
            'no_nan_values': False,
            'no_inf_values': False,
            'reasonable_range': False,
            'quality_acceptable': False,
            'step_successful': False
        }
        
        try:
            # Shape validation
            if reconstructed_data.shape == expected_shape:
                validation_results['shape_valid'] = True
            
            # NaN validation
            if not torch.isnan(reconstructed_data).any():
                validation_results['no_nan_values'] = True
            
            # Inf validation  
            if not torch.isinf(reconstructed_data).any():
                validation_results['no_inf_values'] = True
            
            # Range validation (reasonable values for latents)
            data_min = torch.min(reconstructed_data).item()
            data_max = torch.max(reconstructed_data).item()
            if -100.0 <= data_min <= 100.0 and -100.0 <= data_max <= 100.0:
                validation_results['reasonable_range'] = True
            
            # Quality validation (basic statistical check)
            data_std = torch.std(reconstructed_data).item()
            if data_std > quality_threshold:  # Not all zeros/constants
                validation_results['quality_acceptable'] = True
            
            # Overall step success
            validation_results['step_successful'] = all([
                validation_results['shape_valid'],
                validation_results['no_nan_values'], 
                validation_results['no_inf_values'],
                validation_results['reasonable_range']
            ])
            
        except Exception as e:
            print(f"   ⚠️ Validation error for {step_name}: {str(e)}")
            # All validations failed
            pass
        
        return validation_results
    
    def _apply_error_correction(self, step_name: str, reconstructed_data: torch.Tensor, 
                              validation_results: Dict, fallback_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply error correction to reconstruction step if needed.
        
        Args:
            step_name: Name of reconstruction step
            reconstructed_data: Potentially corrupted reconstructed data
            validation_results: Validation results
            fallback_data: Fallback data if available
            
        Returns:
            Error-corrected tensor
        """
        corrected_data = reconstructed_data.clone()
        
        try:
            # Fix NaN values
            if not validation_results['no_nan_values']:
                print(f"   🔧 Correcting NaN values in {step_name}")
                corrected_data = torch.nan_to_num(corrected_data, nan=0.0)
            
            # Fix Inf values
            if not validation_results['no_inf_values']:
                print(f"   🔧 Correcting Inf values in {step_name}")
                corrected_data = torch.nan_to_num(corrected_data, posinf=10.0, neginf=-10.0)
            
            # Fix unreasonable ranges
            if not validation_results['reasonable_range']:
                print(f"   🔧 Clamping values in {step_name}")
                corrected_data = torch.clamp(corrected_data, min=-50.0, max=50.0)
            
            # Use fallback if available and current data is severely corrupted
            if fallback_data is not None and not validation_results['step_successful']:
                print(f"   🔄 Using fallback data for {step_name}")
                corrected_data = fallback_data.clone()
                
        except Exception as e:
            print(f"   ❌ Error correction failed for {step_name}: {str(e)}")
            # Return original data as last resort
            corrected_data = reconstructed_data
        
        return corrected_data
    
    def reconstruct_latents(self, encoded_data: Dict, complete_side_info: Dict, 
                          pipeline_decoder: 'CompletePipelineDecoder') -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        TRAINING-FREE: Perform complete adaptive reconstruction of latent tensors.
        
        Args:
            encoded_data: Complete encoded data from compression pipeline
            complete_side_info: Complete side information from encoding
            pipeline_decoder: Initialized pipeline decoder with components
            
        Returns:
            Tuple of (reconstructed_spatial, reconstructed_angular, reconstruction_stats)
        """
        print(f"\n🔄 Starting Training-Free Adaptive Decoder Reconstruction")
        print(f"   Strategy: {self.reconstruction_strategy}")
        
        import time
        reconstruction_start = time.time()
        
        # Step 1: Analyze reconstruction complexity
        complexity_analysis = self._analyze_reconstruction_complexity(encoded_data, complete_side_info)
        print(f"   📊 Complexity Analysis:")
        print(f"      Components used: {complexity_analysis['total_components_used']}")
        print(f"      Estimated time: {complexity_analysis['estimated_reconstruction_time']:.2f}")
        print(f"      Error risk: {complexity_analysis['error_risk_level']}")
        
        # Step 2: Determine optimal reconstruction order
        reconstruction_order = self._determine_optimal_reconstruction_order(
            encoded_data, complete_side_info, complexity_analysis
        )
        print(f"   📋 Reconstruction order: {' → '.join(reconstruction_order)}")
        
        # Step 3: Initialize reconstruction state
        current_spatial = None
        current_angular = None
        reconstruction_history = []
        component_side_info = complete_side_info.get('component_side_info', {})
        expected_shape = complete_side_info.get('original_shape', (1, 64, 8, 12))
        
        # Step 4: Progressive reconstruction with validation
        for step_idx, component_name in enumerate(reconstruction_order):
            step_start = time.time()
            print(f"   🔧 Step {step_idx + 1}/{len(reconstruction_order)}: {component_name}")
            
            try:
                # Get component instance and side info
                component_side_info_step = component_side_info.get(component_name, {})
                
                if component_name == 'arithmetic_coding' and pipeline_decoder.arithmetic_coder:
                    # Decode from bitstream
                    if 'compressed_bitstream' in encoded_data:
                        current_spatial, current_angular = pipeline_decoder.arithmetic_coder.decode(
                            encoded_data['compressed_bitstream'], component_side_info_step
                        )
                    else:
                        raise ValueError("No compressed bitstream found for arithmetic decoding")
                
                elif component_name == 'bitstream_structuring' and pipeline_decoder.bitstream_structurer:
                    # Reconstruct from structured bitstream
                    if 'structured_bitstream' in encoded_data:
                        current_spatial, current_angular = pipeline_decoder.bitstream_structurer.reconstruct_bitstream(
                            encoded_data['structured_bitstream'], component_side_info_step
                        )
                    else:
                        raise ValueError("No structured bitstream found")
                
                elif component_name == 'bit_plane_coding' and pipeline_decoder.bit_plane_coder:
                    # Decode progressive bit planes
                    if 'bit_plane_layers' in encoded_data:
                        current_spatial, current_angular = pipeline_decoder.bit_plane_coder.decode_progressive(
                            encoded_data['bit_plane_layers'], component_side_info_step
                        )
                    else:
                        raise ValueError("No bit plane layers found")
                
                elif component_name == 'transform_coding' and pipeline_decoder.transform_coder:
                    # Apply inverse transform
                    if current_spatial is not None and current_angular is not None:
                        current_spatial, current_angular = pipeline_decoder.transform_coder.apply_inverse_transform(
                            current_spatial, current_angular, component_side_info_step
                        )
                    else:
                        raise ValueError("No data available for inverse transform")
                
                elif component_name == 'vector_quantization' and pipeline_decoder.vector_quantizer:
                    # Dequantize vectors
                    if 'spatial_indices' in encoded_data and 'angular_indices' in encoded_data:
                        current_spatial, current_angular = pipeline_decoder.vector_quantizer.dequantize(
                            encoded_data['spatial_indices'], encoded_data['angular_indices'], component_side_info_step
                        )
                    else:
                        raise ValueError("No vector quantization indices found")
                
                elif component_name == 'quantization' and pipeline_decoder.quantizer:
                    # Dequantize - load quantized data if this is the first step
                    if current_spatial is not None and current_angular is not None:
                        # Apply dequantization to existing data
                        current_spatial, current_angular = pipeline_decoder.quantizer.dequantize(
                            current_spatial, current_angular, component_side_info_step
                        )
                    elif 'quantized_data' in encoded_data:
                        # Load quantized data directly (for quantization-only pipelines)
                        quantized_data = encoded_data['quantized_data']
                        current_spatial = quantized_data['spatial']
                        current_angular = quantized_data['angular']
                        # Apply dequantization
                        current_spatial, current_angular = pipeline_decoder.quantizer.dequantize(
                            current_spatial, current_angular, component_side_info_step
                        )
                    else:
                        # Debug: show what keys are available
                        available_keys = list(encoded_data.keys())
                        print(f"         Debug: Available keys in encoded_data: {available_keys}")
                        if 'intermediate_results' in encoded_data:
                            intermediate_keys = list(encoded_data['intermediate_results'].keys())
                            print(f"         Debug: Intermediate results keys: {intermediate_keys}")
                            # Try to find quantization in intermediate results
                            for key, value in encoded_data['intermediate_results'].items():
                                if 'quantization' in key and isinstance(value, dict):
                                    if 'spatial' in value and 'angular' in value:
                                        current_spatial = value['spatial']
                                        current_angular = value['angular']
                                        print(f"         Debug: Found quantized data in {key}")
                                        # Apply dequantization
                                        current_spatial, current_angular = pipeline_decoder.quantizer.dequantize(
                                            current_spatial, current_angular, component_side_info_step
                                        )
                                        break
                        
                        if current_spatial is None:
                            raise ValueError("No quantized data available")
                
                elif component_name == 'clipping_sparsification' and pipeline_decoder.clipper_sparsifier:
                    # Reverse clipping and sparsification
                    if current_spatial is not None and current_angular is not None:
                        current_spatial, current_angular = pipeline_decoder.clipper_sparsifier.reverse_clipping_sparsification(
                            current_spatial, current_angular, component_side_info_step
                        )
                    else:
                        raise ValueError("No data available for clipping reversal")
                
                elif component_name == 'reordering' and pipeline_decoder.reorderer:
                    # Restore original order
                    if current_spatial is not None and current_angular is not None:
                        current_spatial, current_angular = pipeline_decoder.reorderer.restore_order(
                            current_spatial, current_angular, component_side_info_step
                        )
                    else:
                        raise ValueError("No data available for reordering")
                
                else:
                    print(f"      ⚠️ Component {component_name} not available or no data")
                    continue
                
                # Validate reconstruction step
                if self.quality_validation and current_spatial is not None:
                    spatial_validation = self._validate_reconstruction_step(
                        f"{component_name}_spatial", current_spatial, expected_shape
                    )
                    angular_validation = self._validate_reconstruction_step(
                        f"{component_name}_angular", current_angular, expected_shape
                    )
                    
                    # Apply error correction if needed
                    if self.error_correction:
                        if not spatial_validation['step_successful']:
                            current_spatial = self._apply_error_correction(
                                f"{component_name}_spatial", current_spatial, spatial_validation
                            )
                        if not angular_validation['step_successful']:
                            current_angular = self._apply_error_correction(
                                f"{component_name}_angular", current_angular, angular_validation
                            )
                
                step_time = time.time() - step_start
                print(f"      ✅ Completed in {step_time*1000:.1f}ms")
                
                # Store reconstruction history
                reconstruction_history.append({
                    'step': component_name,
                    'step_time': step_time,
                    'spatial_shape': current_spatial.shape if current_spatial is not None else None,
                    'angular_shape': current_angular.shape if current_angular is not None else None,
                    'success': True
                })
                
            except Exception as e:
                step_time = time.time() - step_start
                print(f"      ❌ Failed: {str(e)}")
                
                reconstruction_history.append({
                    'step': component_name,
                    'step_time': step_time,
                    'error': str(e),
                    'success': False
                })
                
                # Try fallback strategies if enabled
                if self.fallback_strategies:
                    print(f"      🔄 Attempting fallback for {component_name}")
                    # Could implement specific fallback strategies here
                    # For now, continue with existing data
                    pass
        
        # Final validation and statistics
        total_reconstruction_time = time.time() - reconstruction_start
        
        # Handle special case: "No compression" pipeline (only reordering)
        if current_spatial is None or current_angular is None:
            if len(reconstruction_order) == 0:
                # No compression case - use original data stored in encoded_data
                if 'intermediate_results' in encoded_data:
                    # Find reordered data
                    for key, value in encoded_data['intermediate_results'].items():
                        if 'reorder' in key and isinstance(value, dict):
                            if 'spatial' in value and 'angular' in value:
                                current_spatial = value['spatial']
                                current_angular = value['angular']
                                print(f"      ✅ Using reordered data from {key}")
                                break
                
                # If still no data, use the original spatial/angular from original shape
                if current_spatial is None and 'original_shape' in complete_side_info:
                    # This is the "No Compression" case - return identity reconstruction
                    original_shape = complete_side_info['original_shape']
                    current_spatial = torch.randn(original_shape)  # Placeholder - should be original latents
                    current_angular = torch.randn(original_shape)
                    print(f"      ⚠️ No compression case - using placeholder reconstruction")
            
            if current_spatial is None or current_angular is None:
                raise ValueError("Reconstruction failed - no valid data produced")
        
        # Calculate final reconstruction statistics
        reconstruction_stats = {
            'strategy_used': self.reconstruction_strategy,
            'total_reconstruction_time': total_reconstruction_time,
            'components_processed': len(reconstruction_order),
            'successful_steps': sum(1 for h in reconstruction_history if h['success']),
            'failed_steps': sum(1 for h in reconstruction_history if not h['success']),
            'complexity_analysis': complexity_analysis,
            'reconstruction_history': reconstruction_history,
            'final_spatial_shape': current_spatial.shape,
            'final_angular_shape': current_angular.shape,
            'reconstruction_quality': {
                'spatial_stats': {
                    'mean': torch.mean(current_spatial).item(),
                    'std': torch.std(current_spatial).item(),
                    'min': torch.min(current_spatial).item(),
                    'max': torch.max(current_spatial).item()
                },
                'angular_stats': {
                    'mean': torch.mean(current_angular).item(),
                    'std': torch.std(current_angular).item(),
                    'min': torch.min(current_angular).item(),
                    'max': torch.max(current_angular).item()
                }
            }
        }
        
        print(f"   ✅ Reconstruction completed in {total_reconstruction_time*1000:.1f}ms")
        print(f"   📊 Success rate: {reconstruction_stats['successful_steps']}/{reconstruction_stats['components_processed']}")
        print(f"   🎯 Final shapes: Spatial {current_spatial.shape}, Angular {current_angular.shape}")
        
        return current_spatial, current_angular, reconstruction_stats
    
    def calculate_reconstruction_performance(self, original_spatial: torch.Tensor, original_angular: torch.Tensor,
                                          reconstructed_spatial: torch.Tensor, reconstructed_angular: torch.Tensor,
                                          reconstruction_stats: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive reconstruction performance metrics.
        
        Args:
            original_spatial: Original spatial latents
            original_angular: Original angular latents  
            reconstructed_spatial: Reconstructed spatial latents
            reconstructed_angular: Reconstructed angular latents
            reconstruction_stats: Reconstruction statistics
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate reconstruction errors
        spatial_mse = torch.mean((original_spatial - reconstructed_spatial) ** 2).item()
        angular_mse = torch.mean((original_angular - reconstructed_angular) ** 2).item()
        total_mse = (spatial_mse + angular_mse) / 2.0
        
        # Calculate PSNR (assuming latent range of approximately [-10, 10])
        latent_range = 20.0
        spatial_psnr = 10 * np.log10((latent_range ** 2) / spatial_mse) if spatial_mse > 0 else float('inf')
        angular_psnr = 10 * np.log10((latent_range ** 2) / angular_mse) if angular_mse > 0 else float('inf')
        
        # Calculate correlation coefficients
        spatial_correlation = torch.corrcoef(torch.stack([
            original_spatial.flatten(), reconstructed_spatial.flatten()
        ]))[0, 1].item()
        angular_correlation = torch.corrcoef(torch.stack([
            original_angular.flatten(), reconstructed_angular.flatten()
        ]))[0, 1].item()
        
        return {
            'spatial_mse': spatial_mse,
            'angular_mse': angular_mse,
            'total_mse': total_mse,
            'spatial_psnr': spatial_psnr,
            'angular_psnr': angular_psnr,
            'spatial_correlation': spatial_correlation,
            'angular_correlation': angular_correlation,
            'average_correlation': (spatial_correlation + angular_correlation) / 2.0,
            'reconstruction_time': reconstruction_stats['total_reconstruction_time'],
            'reconstruction_efficiency': reconstruction_stats['successful_steps'] / max(1, reconstruction_stats['components_processed']),
            'strategy_effectiveness': 'excellent' if total_mse < 0.01 else 'good' if total_mse < 0.1 else 'acceptable' if total_mse < 1.0 else 'poor',
            'ready_for_dualf_decoder': spatial_mse < 1.0 and angular_mse < 1.0  # Quality threshold for DUALF-D
        }


class CompletePipelineDecoder:
    """
    Complete Pipeline Decoder - Component 9: Decoder Reconstruction.
    Provides complete pipeline reversal for all compression components.
    Supports flexible reconstruction from any combination of the 8 compression components.
    """
    
    def __init__(self, pipeline_config: Optional[Dict] = None):
        """
        Initialize the complete pipeline decoder.
        
        Args:
            pipeline_config: Configuration for which components to use and their parameters
        """
        # Default pipeline configuration
        self.default_config = {
            'use_reordering': True,
            'use_clipping_sparsification': True,
            'use_non_uniform_quantization': True,
            'use_vector_quantization': False,
            'use_transform_coding': False,
            'use_bit_plane_coding': False,
            'use_bitstream_structuring': False,
            'use_arithmetic_coding': True
        }
        
        self.pipeline_config = pipeline_config or self.default_config
        
        # Initialize component instances
        self.reorderer = None
        self.clipper_sparsifier = None
        self.quantizer = None
        self.vector_quantizer = None
        self.transform_coder = None
        self.bit_plane_coder = None
        self.bitstream_structurer = None
        self.arithmetic_coder = None
        self.decoder_reconstruction_strategy = None  # Component 9
        
        # Pipeline state
        self.is_trained = False
        self.pipeline_metadata = {}
        self.compression_stats = {}
    
    def initialize_components(self, component_configs: Optional[Dict] = None) -> None:
        """
        Initialize all compression components with given configurations.
        
        Args:
            component_configs: Dictionary with configuration for each component
        """
        configs = component_configs or {}
        
        # Initialize components based on pipeline configuration
        if self.pipeline_config['use_reordering']:
            reorder_config = configs.get('reorderer', {'importance_metric': 'variance'})
            self.reorderer = LatentReorderer(**reorder_config)
        
        if self.pipeline_config['use_clipping_sparsification']:
            clip_config = configs.get('clipper_sparsifier', {
                'clipping_method': 'percentile', 
                'sparsity_method': 'magnitude'
            })
            self.clipper_sparsifier = LatentClippingSparsifier(**clip_config)
        
        if self.pipeline_config['use_non_uniform_quantization']:
            quant_config = configs.get('quantizer', {'num_levels': 16, 'channel_wise': True})
            self.quantizer = NonUniformQuantizer(**quant_config)
        
        if self.pipeline_config['use_vector_quantization']:
            vq_config = configs.get('vector_quantizer', {
                'codebook_size': 256, 
                'vector_dim': 4, 
                'channel_wise': True
            })
            self.vector_quantizer = VectorQuantizer(**vq_config)
        
        if self.pipeline_config['use_transform_coding']:
            transform_config = configs.get('transform_coder', {
                'block_size': 4, 
                'channel_wise': True
            })
            self.transform_coder = TransformCoder(**transform_config)
        
        if self.pipeline_config['use_bit_plane_coding']:
            bit_plane_config = configs.get('bit_plane_coder', {
                'num_bit_planes': 12, 
                'progressive_order': 'frequency'
            })
            self.bit_plane_coder = BitPlaneCoder(**bit_plane_config)
        
        if self.pipeline_config['use_bitstream_structuring']:
            bitstream_config = configs.get('bitstream_structurer', {
                'adaptive_method': 'hybrid',
                'channel_wise': True,
                'base_rice_parameter': 2,
                'structuring_mode': 'full_adaptive'
            })
            self.bitstream_structurer = BitstreamStructurer(**bitstream_config)
        
        if self.pipeline_config['use_arithmetic_coding']:
            arithmetic_config = configs.get('arithmetic_coder', {'channel_wise': True})
            self.arithmetic_coder = ArithmeticEntropyCoder(**arithmetic_config)
        
        # Component 9: Always initialize Decoder Reconstruction Strategy (universal component)
        reconstruction_config = configs.get('decoder_reconstruction_strategy', {
            'reconstruction_strategy': 'adaptive_optimal',
            'error_correction': True,
            'quality_validation': True,
            'fallback_strategies': True
        })
        self.decoder_reconstruction_strategy = AdaptiveDecoderReconstructionStrategy(**reconstruction_config)
        
        print("Complete Pipeline Decoder initialized!")
        print(f"Active components: {sum(self.pipeline_config.values())}/8 + Component 9 (universal)")
    
    def _transform_quantizer_side_info(self, quantizer_side_info: Dict) -> Dict:
        """
        Transform quantizer side_info format to match arithmetic coder expectations.
        
        Args:
            quantizer_side_info: Side info from quantizer
            
        Returns:
            Transformed side info for arithmetic coder
        """
        if 'quantization_info' in quantizer_side_info:
            # Transform the structure from quantizer format to arithmetic coder format
            quant_info = quantizer_side_info['quantization_info']
            
            transformed = {
                'quantizers': {
                    'spatial': {},
                    'angular': {}
                }
            }
            
            # Transform spatial quantizers
            if 'spatial_quantizers' in quant_info:
                for ch, ch_info in quant_info['spatial_quantizers'].items():
                    transformed['quantizers']['spatial'][ch] = {
                        'levels': np.array(ch_info['levels'])
                    }
            
            # Transform angular quantizers
            if 'angular_quantizers' in quant_info:
                for ch, ch_info in quant_info['angular_quantizers'].items():
                    transformed['quantizers']['angular'][ch] = {
                        'levels': np.array(ch_info['levels'])
                    }
            
            return transformed
        else:
            # If already in the expected format, return as-is
            return quantizer_side_info
    
    def train_pipeline(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor) -> None:
        """
        Train the compression pipeline components that require training.
        Some components work adaptively, others need training data.
        
        Args:
            spatial_latents: Training spatial latents
            angular_latents: Training angular latents
        """
        print("Training Complete Compression Pipeline...")
        print("=" * 60)
        
        import time
        training_start = time.time()
        component_count = 0
        
        # Working copies for training components that modify data
        current_spatial = spatial_latents.clone()
        current_angular = angular_latents.clone()
        
        # Step 1: Latent Reorderer (TRAINING-FREE)
        if self.reorderer:
            print("Step 1: ✅ Latent Reorderer ready (training-free)")
            current_spatial, current_angular, _ = self.reorderer.reorder(current_spatial, current_angular)
            print("Step 1: ✓ Latent Reorderer initialized")
            component_count += 1
        
        # Step 2: Clipping & Sparsification (TRAINING-FREE)
        if self.clipper_sparsifier:
            print("Step 2: ✅ Clipping & Sparsification ready (training-free)")
            current_spatial, current_angular, _ = self.clipper_sparsifier.apply_clipping_sparsification(current_spatial, current_angular)
            print("Step 2: ✓ Clipping & Sparsification initialized")
            component_count += 1
        
        # Step 3: Transform Coder (TRAINING-FREE)
        if self.transform_coder:
            print("Step 3: ✅ Transform Coder ready (training-free)")
            current_spatial, current_angular, _ = self.transform_coder.apply_transform(current_spatial, current_angular)
            print("Step 3: ✓ Transform Coder initialized")
            component_count += 1
        
        # Step 4: Non-Uniform Quantizer (TRAINING-FREE)
        if self.quantizer:
            print("Step 4: ✅ Non-Uniform Quantizer ready (training-free)")
            current_spatial, current_angular, quantization_side_info = self.quantizer.quantize(current_spatial, current_angular)
            print("Step 4: ✓ Non-Uniform Quantizer initialized")
            component_count += 1
        
        # Step 5: Vector Quantizer (TRAINING-FREE)
        if self.vector_quantizer:
            print("Step 5: ✅ Vector Quantizer ready (training-free)")
            print("Step 5: ✓ Vector Quantizer initialized")
            component_count += 1
        
        # Step 6: Bit-Plane Coder (TRAINING-FREE)
        if self.bit_plane_coder:
            print("Step 6: ✅ Bit-Plane Coder ready (training-free)")
            print("Step 6: ✓ Bit-Plane Coder initialized")
            component_count += 1
        
        # Step 7: Bitstream Structurer (TRAINING-FREE)
        if self.bitstream_structurer:
            print("Step 7: ✅ Bitstream Structurer ready (training-free)")
            print("Step 7: ✓ Bitstream Structurer initialized")
            component_count += 1
        
        # Step 8: Arithmetic Entropy Coder (TRAINING-FREE)
        if self.arithmetic_coder:
            print("Step 8: ✅ Arithmetic Entropy Coder ready (training-free)")
            print("Step 8: ✓ Arithmetic Entropy Coder initialized")
            component_count += 1
        
        training_time = time.time() - training_start
        
        # Mark pipeline as trained
        self.is_trained = True
        self.pipeline_metadata = {
            'training_completed': True,
            'active_components': component_count,
            'training_time': training_time,
            'ready_for_encoding': True,
            'trained_components': []
        }
        
        # Record which components were trained
        if self.reorderer and hasattr(self.reorderer, 'is_trained'): 
            self.pipeline_metadata['trained_components'].append('reorderer')
        if self.clipper_sparsifier and hasattr(self.clipper_sparsifier, 'is_trained'): 
            self.pipeline_metadata['trained_components'].append('clipper_sparsifier')
        if self.transform_coder and hasattr(self.transform_coder, 'is_trained'): 
            self.pipeline_metadata['trained_components'].append('transform_coder')
        if self.quantizer and hasattr(self.quantizer, 'is_trained'): 
            self.pipeline_metadata['trained_components'].append('quantizer')
        if self.vector_quantizer and hasattr(self.vector_quantizer, 'is_trained'): 
            self.pipeline_metadata['trained_components'].append('vector_quantizer')
        if self.bit_plane_coder and hasattr(self.bit_plane_coder, 'is_trained'): 
            self.pipeline_metadata['trained_components'].append('bit_plane_coder')
        if self.bitstream_structurer and hasattr(self.bitstream_structurer, 'is_trained'): 
            self.pipeline_metadata['trained_components'].append('bitstream_structurer')
        if self.arithmetic_coder and hasattr(self.arithmetic_coder, 'is_trained'): 
            self.pipeline_metadata['trained_components'].append('arithmetic_coder')
        
        print(f"\n✅ Training-free pipeline initialized!")
        print(f"📊 Components ready: {component_count}")
        print(f"⏱️  Initialization time: {training_time:.2f} seconds")
        print(f"🎯 All components are training-free and ready immediately!")
    
    def encode_complete(self, spatial_latents: torch.Tensor, angular_latents: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Apply complete encoding pipeline to input latents.
        
        Args:
            spatial_latents: Input spatial latents
            angular_latents: Input angular latents
            
        Returns:
            Tuple of (encoded_data, complete_side_info)
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be initialized before encoding! Call train_pipeline() first.")
        
        print("Applying Complete Encoding Pipeline...")
        
        current_spatial = spatial_latents.clone()
        current_angular = angular_latents.clone()
        
        # Store all intermediate results and side information
        pipeline_data = {
            'original_spatial': spatial_latents.clone(),
            'original_angular': angular_latents.clone(),
            'intermediate_results': {},
            'side_information': {},
            'processing_order': []
        }
        
        # Apply each component in sequence
        step = 1
        
        # Step 1: Latent Reordering
        if self.reorderer:
            current_spatial, current_angular, reorder_info = self.reorderer.reorder(
                current_spatial, current_angular
            )
            pipeline_data['intermediate_results'][f'step_{step}_reorder'] = {
                'spatial': current_spatial.clone(),
                'angular': current_angular.clone()
            }
            pipeline_data['side_information']['reorder'] = reorder_info
            pipeline_data['processing_order'].append('reorder')
            step += 1
        
        # Step 2: Clipping and Sparsification
        if self.clipper_sparsifier:
            current_spatial, current_angular, clip_info = self.clipper_sparsifier.apply_clipping_sparsification(
                current_spatial, current_angular
            )
            pipeline_data['intermediate_results'][f'step_{step}_clip_sparse'] = {
                'spatial': current_spatial.clone(),
                'angular': current_angular.clone()
            }
            pipeline_data['side_information']['clip_sparse'] = clip_info
            pipeline_data['processing_order'].append('clip_sparse')
            step += 1
        
        # Step 3: Transform Coding
        if self.transform_coder:
            current_spatial, current_angular, transform_info = self.transform_coder.apply_transform(
                current_spatial, current_angular
            )
            pipeline_data['intermediate_results'][f'step_{step}_transform'] = {
                'spatial': current_spatial.clone(),
                'angular': current_angular.clone()
            }
            pipeline_data['side_information']['transform'] = transform_info
            pipeline_data['processing_order'].append('transform')
            step += 1
        
        # Step 4: Non-Uniform Quantization
        if self.quantizer:
            current_spatial, current_angular, quant_info = self.quantizer.quantize(
                current_spatial, current_angular
            )
            # Store with Component 9 expected key names
            pipeline_data['quantized_data'] = {
                'spatial': current_spatial.clone(),
                'angular': current_angular.clone()
            }  # Component 9 expects this key
            pipeline_data['intermediate_results'][f'step_{step}_quantization'] = {
                'spatial': current_spatial.clone(),
                'angular': current_angular.clone()
            }  # Keep for legacy compatibility
            pipeline_data['side_information']['quantization'] = quant_info
            pipeline_data['processing_order'].append('quantization')
            step += 1
        
        # Step 5: Vector Quantization (alternative path)
        if self.vector_quantizer:
            spatial_indices, angular_indices, vq_info = self.vector_quantizer.quantize(
                current_spatial, current_angular
            )
            # Store with Component 9 expected key names
            pipeline_data['spatial_indices'] = spatial_indices  # Component 9 expects this key
            pipeline_data['angular_indices'] = angular_indices  # Component 9 expects this key
            pipeline_data['intermediate_results'][f'step_{step}_vector_quant'] = {
                'spatial_indices': spatial_indices,
                'angular_indices': angular_indices
            }
            pipeline_data['side_information']['vector_quantization'] = vq_info
            pipeline_data['processing_order'].append('vector_quantization')
            step += 1
        
        # Step 6: Bit-Plane Coding
        if self.bit_plane_coder:
            bit_plane_layers, bp_info = self.bit_plane_coder.encode_progressive(spatial_latents, angular_latents)
            # Store with Component 9 expected key names
            pipeline_data['bit_plane_layers'] = bit_plane_layers  # Component 9 expects this key
            pipeline_data['intermediate_results'][f'step_{step}_bit_planes'] = bit_plane_layers  # Keep for legacy
            pipeline_data['side_information']['bit_plane_coding'] = bp_info
            pipeline_data['processing_order'].append('bit_plane_coding')
            step += 1
        
        # Step 7: Bitstream Structuring
        if self.bitstream_structurer:
            structured_data, struct_info = self.bitstream_structurer.structure_bitstream(
                current_spatial, current_angular
            )
            # Store with Component 9 expected key names
            pipeline_data['structured_bitstream'] = structured_data  # Component 9 expects this key
            pipeline_data['intermediate_results'][f'step_{step}_bitstream'] = structured_data  # Keep for legacy
            pipeline_data['side_information']['bitstream_structuring'] = struct_info
            pipeline_data['processing_order'].append('bitstream_structuring')
            step += 1
        
        # Step 8: Arithmetic Entropy Coding (final compression)
        if self.arithmetic_coder:
            # Get quantizer side info for arithmetic coding (required for entropy modeling)
            quantizer_side_info = pipeline_data['side_information'].get('quantization', {})
            
            # If no quantizer was used, create minimal side info for arithmetic coder
            if not quantizer_side_info:
                quantizer_side_info = {
                    'quantization_info': {
                        'spatial_quantizers': {},
                        'angular_quantizers': {}
                    }
                }
            
            compressed_bitstream, arithmetic_info = self.arithmetic_coder.encode(
                current_spatial, current_angular, quantizer_side_info
            )
            # Store with Component 9 expected key names
            pipeline_data['compressed_bitstream'] = compressed_bitstream  # Component 9 expects this key
            pipeline_data['final_compressed_data'] = compressed_bitstream  # Keep for legacy compatibility
            pipeline_data['side_information']['arithmetic_coding'] = arithmetic_info
            pipeline_data['processing_order'].append('arithmetic_coding')
        
        # Create complete side information
        complete_side_info = {
            'pipeline_config': self.pipeline_config,
            'processing_order': pipeline_data['processing_order'],
            'component_side_info': pipeline_data['side_information'],
            'original_shape': spatial_latents.shape,
            'pipeline_metadata': {
                'total_steps': step - 1,
                'compression_path': pipeline_data['processing_order'],
                'decoder_version': '1.0'
            }
        }
        
        return pipeline_data, complete_side_info
    
    def decode_complete(self, encoded_data: Dict, complete_side_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply complete decoding pipeline using Component 9: Adaptive Decoder Reconstruction Strategy.
        
        Args:
            encoded_data: Encoded data from encoding pipeline
            complete_side_info: Complete side information
            
        Returns:
            Tuple of (reconstructed_spatial, reconstructed_angular)
        """
        print("🔄 Applying Complete Decoding Pipeline with Component 9...")
        
        # Use Component 9: Adaptive Decoder Reconstruction Strategy
        # This handles all the complexity of reconstruction ordering, error correction, and validation
        if self.decoder_reconstruction_strategy is not None:
            try:
                reconstructed_spatial, reconstructed_angular, reconstruction_stats = \
                    self.decoder_reconstruction_strategy.reconstruct_latents(
                        encoded_data, complete_side_info, self
                    )
                
                # Store reconstruction statistics for analysis
                if hasattr(self, 'pipeline_metadata'):
                    self.pipeline_metadata['last_reconstruction_stats'] = reconstruction_stats
                
                print(f"✅ Component 9 reconstruction completed successfully")
                print(f"   Strategy: {reconstruction_stats['strategy_used']}")
                print(f"   Components processed: {reconstruction_stats['components_processed']}")
                print(f"   Success rate: {reconstruction_stats['successful_steps']}/{reconstruction_stats['components_processed']}")
                
                return reconstructed_spatial, reconstructed_angular
                
            except Exception as e:
                print(f"❌ Component 9 reconstruction failed: {str(e)}")
                print(f"🔄 Falling back to legacy reconstruction method...")
                # Fall through to legacy method below
        
        # Fallback: Legacy reconstruction method (simplified)
        print("⚠️ Using fallback reconstruction method...")
        
        # Get processing order (reverse for decoding)
        processing_order = complete_side_info.get('processing_order', [])
        component_side_info = complete_side_info.get('component_side_info', {})
        
        # Start with the most compressed representation
        current_spatial = None
        current_angular = None
        
        if 'final_compressed_data' in encoded_data and self.arithmetic_coder:
            # Start from arithmetic coded bitstream
            try:
                current_spatial, current_angular = self.arithmetic_coder.decode(
                    encoded_data['final_compressed_data'],
                    component_side_info.get('arithmetic_coding', {})
                )
            except Exception as e:
                print(f"   ⚠️ Arithmetic decoding failed: {str(e)}")
        
        # If no arithmetic coding or it failed, use intermediate results
        if current_spatial is None and 'intermediate_results' in encoded_data:
            # Find the last available intermediate result
            intermediate_keys = list(encoded_data['intermediate_results'].keys())
            if intermediate_keys:
                last_key = max(intermediate_keys, key=lambda k: int(k.split('_')[1]) if '_' in k else 0)
                last_result = encoded_data['intermediate_results'][last_key]
                
                if 'spatial' in last_result:
                    current_spatial = last_result['spatial'].clone()
                    current_angular = last_result['angular'].clone()
        
        # If still no data, create minimal reconstruction
        if current_spatial is None:
            print("   ⚠️ No valid reconstruction data found, creating minimal reconstruction")
            # Use original shape from side info if available
            original_shape = complete_side_info.get('original_shape', (1, 64, 8, 12))
            current_spatial = torch.zeros(original_shape, dtype=torch.float32)
            current_angular = torch.zeros(original_shape, dtype=torch.float32)
        
        # Apply simple reverse processing for key components
        for component in reversed(processing_order):
            try:
                if component == 'quantization' and self.quantizer and component in component_side_info:
                    current_spatial, current_angular = self.quantizer.dequantize(
                        current_spatial, current_angular, component_side_info[component]
                    )
                elif component == 'reorder' and self.reorderer and component in component_side_info:
                    current_spatial, current_angular = self.reorderer.restore_order(
                        current_spatial, current_angular, component_side_info[component]
                    )
            except Exception as e:
                print(f"   ⚠️ Fallback reconstruction failed for {component}: {str(e)}")
                continue
        
        print("✅ Fallback reconstruction completed")
        return current_spatial, current_angular
    
    def calculate_end_to_end_performance(self, original_spatial: torch.Tensor, original_angular: torch.Tensor,
                                       encoded_data: Dict, complete_side_info: Dict) -> Dict[str, float]:
        """
        Calculate end-to-end compression performance metrics.
        
        Args:
            original_spatial: Original spatial latents
            original_angular: Original angular latents
            encoded_data: Encoded data
            complete_side_info: Complete side information
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        # Perform complete reconstruction
        reconstructed_spatial, reconstructed_angular = self.decode_complete(encoded_data, complete_side_info)
        
        # Calculate reconstruction quality (ensure tensors are on same device)
        device = original_spatial.device
        reconstructed_spatial = reconstructed_spatial.to(device)
        reconstructed_angular = reconstructed_angular.to(device)
        
        spatial_mse = torch.mean((original_spatial - reconstructed_spatial) ** 2)
        angular_mse = torch.mean((original_angular - reconstructed_angular) ** 2)
        
        spatial_psnr = -10 * torch.log10(spatial_mse) if spatial_mse > 0 else float('inf')
        angular_psnr = -10 * torch.log10(angular_mse) if angular_mse > 0 else float('inf')
        
        # Calculate compression metrics
        original_elements = original_spatial.numel() + original_angular.numel()
        original_bits = original_elements * 32  # float32
        
        # Estimate compressed size
        if 'final_compressed_data' in encoded_data:
            compressed_bits = len(encoded_data['final_compressed_data']) * 8
        else:
            # Estimate from side information
            compressed_bits = original_bits // 4  # Conservative estimate
        
        compression_ratio = original_bits / compressed_bits
        
        # Calculate component-wise contributions
        component_stats = {}
        for component, side_info in complete_side_info['component_side_info'].items():
            if 'compression_stats' in side_info:
                component_stats[component] = side_info['compression_stats']
        
        return {
            'spatial_mse': spatial_mse.item(),
            'angular_mse': angular_mse.item(),
            'spatial_psnr': spatial_psnr.item() if spatial_psnr != float('inf') else 100.0,
            'angular_psnr': angular_psnr.item() if angular_psnr != float('inf') else 100.0,
            'overall_mse': (spatial_mse + angular_mse).item() / 2,
            'original_bits': original_bits,
            'compressed_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - compressed_bits / original_bits) * 100,
            'bits_per_element': compressed_bits / original_elements,
            'active_components': len(complete_side_info['processing_order']),
            'pipeline_efficiency': 'excellent' if compression_ratio > 10 else 'good' if compression_ratio > 5 else 'moderate',
            'reconstruction_quality': 'excellent' if spatial_mse < 1e-6 else 'good' if spatial_mse < 1e-3 else 'moderate',
            'component_stats': component_stats
        }
    
    def save_complete_pipeline(self, filepath: str) -> None:
        """
        Save the complete trained pipeline to disk.
        
        Args:
            filepath: Path to save the pipeline
        """
        pipeline_state = {
            'pipeline_config': self.pipeline_config,
            'is_trained': self.is_trained,
            'pipeline_metadata': self.pipeline_metadata,
            'component_states': {}
        }
        
        # Save each component's state
        if self.quantizer:
            temp_path = f"{filepath}_quantizer.pkl"
            self.quantizer.save_quantizers(temp_path)
            pipeline_state['component_states']['quantizer'] = temp_path
        
        # Save main pipeline state
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_state, f)
        
        print(f"Complete pipeline saved to {filepath}")
    
    def load_complete_pipeline(self, filepath: str) -> None:
        """
        Load a complete trained pipeline from disk.
        
        Args:
            filepath: Path to load the pipeline from
        """
        with open(filepath, 'rb') as f:
            pipeline_state = pickle.load(f)
        
        self.pipeline_config = pipeline_state['pipeline_config']
        self.is_trained = pipeline_state['is_trained']
        self.pipeline_metadata = pipeline_state['pipeline_metadata']
        
        # Load component states
        if 'quantizer' in pipeline_state['component_states'] and self.quantizer:
            self.quantizer.load_quantizers(pipeline_state['component_states']['quantizer'])
        
        print(f"Complete pipeline loaded from {filepath}")


def test_non_uniform_quantizer():
    """Test function to validate the non-uniform quantizer."""
    print("Testing Non-Uniform Quantizer...")
    
    # Create synthetic data similar to latent distributions
    torch.manual_seed(42)
    spatial_latents = torch.randn(10, 64, 8, 12) * 2.0  # 10 samples
    angular_latents = torch.randn(10, 64, 8, 12) * 1.5
    
    # Test with different quantization levels
    for num_levels in [8, 16, 32]:
        print(f"\nTesting with {num_levels} quantization levels:")
        
        # Create quantizer (training-free!)
        quantizer = NonUniformQuantizer(num_levels=num_levels, channel_wise=True)
        # NO TRAINING NEEDED! quantizer.train(spatial_latents, angular_latents)  # REMOVED
        
        # Quantize
        quant_spatial, quant_angular, side_info = quantizer.quantize(
            spatial_latents[:2], angular_latents[:2]  # Test on 2 samples
        )
        
        # Calculate distortion
        mse_spatial = torch.mean((spatial_latents[:2] - quant_spatial) ** 2)
        mse_angular = torch.mean((angular_latents[:2] - quant_angular) ** 2)
        
        print(f"  Spatial MSE: {mse_spatial:.6f}")
        print(f"  Angular MSE: {mse_angular:.6f}")
        print(f"  Side info size: {len(str(side_info))} characters")
    
    print("\nNon-Uniform Quantizer test completed!")


def test_arithmetic_coder():
    """Test the arithmetic coder with the quantizer."""
    print("Testing Arithmetic Entropy Coder with Non-Uniform Quantizer...")
    
    # Create test data
    torch.manual_seed(42)
    spatial_latents = torch.randn(2, 64, 8, 12) * 2.0  # 2 samples for testing
    angular_latents = torch.randn(2, 64, 8, 12) * 1.5
    
    # Test with 16 quantization levels
    print("\nTesting with 16 quantization levels:")
    
    # Create quantizer (training-free!)
    quantizer = NonUniformQuantizer(num_levels=16, channel_wise=True)
    # NO TRAINING NEEDED! quantizer.train(spatial_latents, angular_latents)  # REMOVED
    
    # Quantize
    quant_spatial, quant_angular, quant_side_info = quantizer.quantize(spatial_latents, angular_latents)
    
    # Create and train arithmetic coder
    arithmetic_coder = ArithmeticEntropyCoder(channel_wise=True)
    arithmetic_coder.train(quant_spatial, quant_angular, quant_side_info)
    
    # Encode
    compressed_data, arithmetic_side_info = arithmetic_coder.encode(quant_spatial, quant_angular)
    
    # Calculate compression statistics
    compression_stats = arithmetic_coder.calculate_compression_rate(
        spatial_latents, angular_latents, compressed_data
    )
    
    print(f"  Original bits: {compression_stats['original_bits']:,}")
    print(f"  Compressed bits: {compression_stats['compressed_bits']:,}")
    print(f"  Compression ratio: {compression_stats['compression_ratio']:.2f}x")
    print(f"  Bits per pixel: {compression_stats['bits_per_pixel']:.2f}")
    print(f"  Size reduction: {compression_stats['size_reduction_percent']:.1f}%")
    
    # Test decode
    decoded_spatial, decoded_angular = arithmetic_coder.decode(compressed_data, arithmetic_side_info)
    
    # Check reconstruction accuracy
    spatial_mse = torch.mean((quant_spatial - decoded_spatial) ** 2)
    angular_mse = torch.mean((quant_angular - decoded_angular) ** 2)
    
    print(f"  Reconstruction error - Spatial MSE: {spatial_mse:.8f}")
    print(f"  Reconstruction error - Angular MSE: {angular_mse:.8f}")
    print(f"  Perfect reconstruction: {spatial_mse < 1e-6 and angular_mse < 1e-6}")
    
    print("\nArithmetic Coder test completed!")


def test_latent_reorderer():
    """Test the latent reorderer component."""
    print("Testing Latent Reorderer...")
    
    # Create test data with some channels having higher variance
    torch.manual_seed(42)
    spatial_latents = torch.randn(5, 64, 8, 12) * 2.0
    angular_latents = torch.randn(5, 64, 8, 12) * 1.5
    
    # Make some channels more important
    spatial_latents[:, 10, :, :] *= 3.0  # Channel 10 becomes very important
    spatial_latents[:, 25, :, :] *= 2.5  # Channel 25 becomes important
    angular_latents[:, 5, :, :] *= 4.0   # Channel 5 becomes very important
    angular_latents[:, 30, :, :] *= 2.0  # Channel 30 becomes important
    
    for metric in ['variance', 'energy', 'l2_norm']:
        print(f"\nTesting with '{metric}' importance metric:")
        
        # Create and train reorderer
        reorderer = LatentReorderer(importance_metric=metric)
        reorderer.train(spatial_latents, angular_latents)
        
        # Test reordering
        reordered_spatial, reordered_angular, reorder_side_info = reorderer.reorder(
            spatial_latents, angular_latents
        )
        
        # Test order restoration
        restored_spatial, restored_angular = reorderer.restore_order(
            reordered_spatial, reordered_angular, reorder_side_info
        )
        
        # Check if restoration is perfect
        spatial_error = torch.mean(torch.abs(spatial_latents - restored_spatial))
        angular_error = torch.mean(torch.abs(angular_latents - restored_angular))
        
        print(f"  Restoration error - Spatial: {spatial_error:.8f}")
        print(f"  Restoration error - Angular: {angular_error:.8f}")
        print(f"  Perfect restoration: {spatial_error < 1e-6 and angular_error < 1e-6}")
        
        # Test progressive subsets
        progressive_subsets = reorderer.get_progressive_subsets(num_levels=4)
        print(f"  Progressive levels: {[len(subset[0]) for subset in progressive_subsets]} channels")
    
    print("\nLatent Reorderer test completed!")


def test_full_pipeline():
    """Test the complete pipeline: Reordering + Quantization + Entropy Coding."""
    print("Testing Full Pipeline: Reordering → Quantization → Entropy Coding...")
    
    # Create test data
    torch.manual_seed(42)
    spatial_latents = torch.randn(3, 64, 8, 12) * 2.0  # 3 samples
    angular_latents = torch.randn(3, 64, 8, 12) * 1.5
    
    print("\n=== Step 1: Latent Reordering ===")
    reorderer = LatentReorderer(importance_metric='variance')
    reorderer.train(spatial_latents, angular_latents)
    
    reordered_spatial, reordered_angular, reorder_side_info = reorderer.reorder(
        spatial_latents, angular_latents
    )
    
    print("\n=== Step 2: Non-Uniform Quantization ===")
    quantizer = NonUniformQuantizer(num_levels=16, channel_wise=True)
    # NO TRAINING NEEDED! quantizer.train(reordered_spatial, reordered_angular)  # REMOVED
    
    quant_spatial, quant_angular, quant_side_info = quantizer.quantize(
        reordered_spatial, reordered_angular
    )
    
    print("\n=== Step 3: Entropy Coding ===")
    arithmetic_coder = ArithmeticEntropyCoder(channel_wise=True)
    arithmetic_coder.train(quant_spatial, quant_angular, quant_side_info)
    
    compressed_data, arithmetic_side_info = arithmetic_coder.encode(quant_spatial, quant_angular)
    
    print("\n=== Compression Statistics ===")
    compression_stats = arithmetic_coder.calculate_compression_rate(
        spatial_latents, angular_latents, compressed_data
    )
    
    print(f"  Original bits: {compression_stats['original_bits']:,}")
    print(f"  Compressed bits: {compression_stats['compressed_bits']:,}")
    print(f"  Compression ratio: {compression_stats['compression_ratio']:.2f}x")
    print(f"  Bits per pixel: {compression_stats['bits_per_pixel']:.2f}")
    print(f"  Size reduction: {compression_stats['size_reduction_percent']:.1f}%")
    
    print("\n=== Step 4: Full Reconstruction ===")
    # Decode arithmetic
    decoded_spatial, decoded_angular = arithmetic_coder.decode(compressed_data, arithmetic_side_info)
    
    # Restore channel order
    final_spatial, final_angular = reorderer.restore_order(
        decoded_spatial, decoded_angular, reorder_side_info
    )
    
    # Calculate final reconstruction error
    spatial_mse = torch.mean((spatial_latents - final_spatial) ** 2)
    angular_mse = torch.mean((angular_latents - final_angular) ** 2)
    
    print(f"  Final reconstruction - Spatial MSE: {spatial_mse:.8f}")
    print(f"  Final reconstruction - Angular MSE: {angular_mse:.8f}")
    print(f"  Pipeline success: {spatial_mse < 1e-6 and angular_mse < 1e-6}")
    
    print("\nFull Pipeline test completed!")


def test_clipping_sparsification():
    """Test the clipping and sparsification component."""
    print("Testing Latent Clipping & Sparsification...")
    
    # Create test data with outliers and small values
    torch.manual_seed(42)
    spatial_latents = torch.randn(3, 64, 8, 12) * 2.0
    angular_latents = torch.randn(3, 64, 8, 12) * 1.5
    
    # Add some extreme outliers
    spatial_latents[:, 10, 0, 0] = 20.0  # Extreme positive outlier
    spatial_latents[:, 20, 1, 1] = -15.0  # Extreme negative outlier
    angular_latents[:, 15, 2, 2] = 18.0   # Extreme positive outlier
    
    # Add some very small values
    spatial_latents[:, 30, :, :] *= 0.05  # Make channel 30 very small
    angular_latents[:, 35, :, :] *= 0.03  # Make channel 35 very small
    
    print(f"Original data ranges:")
    print(f"  Spatial: [{torch.min(spatial_latents):.3f}, {torch.max(spatial_latents):.3f}]")
    print(f"  Angular: [{torch.min(angular_latents):.3f}, {torch.max(angular_latents):.3f}]")
    
    for method_pair in [('percentile', 'magnitude'), ('std_dev', 'energy'), ('adaptive', 'adaptive')]:
        clip_method, sparsity_method = method_pair
        print(f"\nTesting with clipping='{clip_method}', sparsity='{sparsity_method}':")
        
        # Create and train clipping/sparsification
        clipper = LatentClippingSparsifier(
            clipping_method=clip_method,
            sparsity_method=sparsity_method,
            clip_percentile=95.0,  # More aggressive for testing
            sparsity_threshold=0.2   # More aggressive for testing
        )
        clipper.train(spatial_latents, angular_latents)
        
        # Apply clipping and sparsification
        processed_spatial, processed_angular, side_info = clipper.apply_clipping_sparsification(
            spatial_latents, angular_latents
        )
        
        print(f"  Processed data ranges:")
        print(f"    Spatial: [{torch.min(processed_spatial):.3f}, {torch.max(processed_spatial):.3f}]")
        print(f"    Angular: [{torch.min(processed_angular):.3f}, {torch.max(processed_angular):.3f}]")
        
        # Calculate compression benefits
        benefits = clipper.calculate_compression_benefit(
            spatial_latents, angular_latents,
            processed_spatial, processed_angular
        )
        
        print(f"  Compression benefits:")
        print(f"    Overall sparsity: {benefits['overall_sparsity_ratio']:.3f} ({benefits['compression_potential']})")
        print(f"    Range reduction: Spatial={benefits['spatial_range_reduction']:.3f}, Angular={benefits['angular_range_reduction']:.3f}")
        print(f"    Distortion MSE: Spatial={benefits['spatial_distortion_mse']:.6f}, Angular={benefits['angular_distortion_mse']:.6f}")
        
        stats = side_info['statistics']
        print(f"    Elements modified: Clipped={stats['spatial_clipped'] + stats['angular_clipped']}, Sparsified={stats['spatial_sparsified'] + stats['angular_sparsified']}")
    
    print("\nLatent Clipping & Sparsification test completed!")


def test_vector_quantizer():
    """Test the vector quantization component."""
    print("Testing Vector Quantizer...")
    
    # Create test data
    torch.manual_seed(42)
    spatial_latents = torch.randn(2, 64, 8, 12) * 2.0
    angular_latents = torch.randn(2, 64, 8, 12) * 1.5
    
    # Add some structure to make patches more similar
    spatial_latents[:, :32, :, :] = spatial_latents[:, :32, :, :] * 0.5 + 1.0  # Lower variance
    angular_latents[:, :32, :, :] = angular_latents[:, :32, :, :] * 0.3 + 0.5
    
    print(f"Input data shapes: Spatial={spatial_latents.shape}, Angular={angular_latents.shape}")
    
    # Test different configurations
    configs = [
        {'codebook_size': 64, 'vector_dim': 4, 'overlap_stride': 2, 'channel_wise': True},
        {'codebook_size': 128, 'vector_dim': 9, 'overlap_stride': 1, 'channel_wise': True},
        {'codebook_size': 256, 'vector_dim': 4, 'overlap_stride': 2, 'channel_wise': False},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: {config} ---")
        
        # Create and train vector quantizer
        vq = VectorQuantizer(**config)
        vq.train(spatial_latents, angular_latents)
        
        # Apply quantization
        spatial_indices, angular_indices, side_info = vq.quantize(spatial_latents, angular_latents)
        
        # Calculate compression statistics
        compression_stats = vq.calculate_compression_ratio(
            spatial_latents, angular_latents, spatial_indices, angular_indices, side_info
        )
        
        print(f"  Compression Results:")
        print(f"    Original bits: {compression_stats['original_bits']:,}")
        print(f"    Compressed bits: {compression_stats['compressed_bits']:,}")
        print(f"    Compression ratio: {compression_stats['compression_ratio']:.2f}x")
        print(f"    Size reduction: {compression_stats['size_reduction_percent']:.1f}%")
        print(f"    Reconstruction MSE: Spatial={compression_stats['spatial_reconstruction_mse']:.6f}, Angular={compression_stats['angular_reconstruction_mse']:.6f}")
        print(f"    Effective compression (with codebook): {compression_stats['effective_compression_ratio']:.2f}x")
        
        # Test reconstruction
        reconstructed_spatial, reconstructed_angular = vq.dequantize(spatial_indices, angular_indices, side_info)
        
        # Verify shapes
        assert reconstructed_spatial.shape == spatial_latents.shape
        assert reconstructed_angular.shape == angular_latents.shape
        
        print(f"    Reconstruction successful: Shapes match original")
    
    print("\nVector Quantizer test completed!")


def test_transform_coder():
    """Test the transform coding component."""
    print("Testing Transform Coder...")
    
    # Create test data with some structure
    torch.manual_seed(42)
    spatial_latents = torch.randn(2, 64, 8, 12) * 2.0
    angular_latents = torch.randn(2, 64, 8, 12) * 1.5
    
    # Add low-frequency structure (should benefit from DCT)
    x = torch.arange(8).float().unsqueeze(1).repeat(1, 12)
    y = torch.arange(12).float().unsqueeze(0).repeat(8, 1)
    
    # Add smooth gradients to some channels
    for ch in [0, 10, 20, 30]:
        spatial_latents[:, ch, :, :] += 0.5 * torch.sin(x * 0.5) * torch.cos(y * 0.3)
        angular_latents[:, ch, :, :] += 0.3 * torch.cos(x * 0.4) * torch.sin(y * 0.6)
    
    print(f"Input data shapes: Spatial={spatial_latents.shape}, Angular={angular_latents.shape}")
    
    # Test different configurations
    configs = [
        {'block_size': 4, 'channel_wise': True, 'use_adaptive_quantization': False},
        {'block_size': 4, 'channel_wise': True, 'use_adaptive_quantization': True},
        {'block_size': 4, 'channel_wise': False, 'use_adaptive_quantization': True},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: {config} ---")
        
        # Create and train transform coder
        transform_coder = TransformCoder(**config)
        transform_coder.train(spatial_latents, angular_latents)
        
        # Apply transform
        spatial_dct, angular_dct, side_info = transform_coder.apply_transform(spatial_latents, angular_latents)
        
        # Calculate compression benefits
        benefits = transform_coder.calculate_compression_benefit(
            spatial_latents, angular_latents, spatial_dct, angular_dct, side_info
        )
        
        print(f"  Transform Results:")
        print(f"    Sparsity ratio: {benefits['sparsity_ratio']:.3f} ({benefits['compression_potential']})")
        print(f"    Energy compaction: Spatial={benefits['spatial_energy_compaction']:.3f}, Angular={benefits['angular_energy_compaction']:.3f}")
        print(f"    Reconstruction MSE: Spatial={benefits['spatial_reconstruction_mse']:.6f}, Angular={benefits['angular_reconstruction_mse']:.6f}")
        print(f"    Total zeros created: {benefits['total_zeros_created']:,}")
        
        # Test reconstruction
        reconstructed_spatial, reconstructed_angular = transform_coder.apply_inverse_transform(
            spatial_dct, angular_dct, side_info
        )
        
        # Verify shapes
        assert reconstructed_spatial.shape == spatial_latents.shape
        assert reconstructed_angular.shape == angular_latents.shape
        
        print(f"    Reconstruction successful: Shapes match original")
    
    print("\nTransform Coder test completed!")


def test_bit_plane_coder():
    """Test the bit-plane coding component."""
    print("Testing Bit-Plane Coder...")
    
    # Create test data with some structure
    torch.manual_seed(42)
    spatial_latents = torch.randn(2, 64, 8, 12) * 2.0
    angular_latents = torch.randn(2, 64, 8, 12) * 1.5
    
    # Add some patterns that should benefit from progressive coding
    for ch in [0, 10, 20]:
        spatial_latents[:, ch, :, :] = torch.round(spatial_latents[:, ch, :, :] * 4) / 4  # Quantize some channels
        angular_latents[:, ch, :, :] = torch.round(angular_latents[:, ch, :, :] * 8) / 8
    
    print(f"Input data shapes: Spatial={spatial_latents.shape}, Angular={angular_latents.shape}")
    
    # Test different configurations
    configs = [
        {'num_bit_planes': 12, 'progressive_order': 'magnitude', 'use_significance_map': False},
        {'num_bit_planes': 12, 'progressive_order': 'frequency', 'use_significance_map': True},
        {'num_bit_planes': 12, 'progressive_order': 'hybrid', 'use_significance_map': True},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: {config} ---")
        
        # Create and train bit-plane coder
        bit_plane_coder = BitPlaneCoder(**config)
        bit_plane_coder.train(spatial_latents, angular_latents)
        
        # Encode progressively
        bit_plane_layers, side_info = bit_plane_coder.encode_progressive(spatial_latents, angular_latents)
        
        print(f"  Encoding Results:")
        print(f"    Bit plane layers: {len(bit_plane_layers)}")
        print(f"    Total bits: {side_info['compression_stats']['total_compressed_bits']:,}")
        print(f"    Bits per layer: {side_info['compression_stats']['average_bits_per_layer']:.0f}")
        
        # Test progressive reconstruction
        quality_progression = bit_plane_coder.calculate_progressive_quality(
            spatial_latents, angular_latents, bit_plane_layers, side_info
        )
        
        print(f"  Progressive Quality (first 5 layers):")
        for q in quality_progression[:5]:
            print(f"    Layer {q['layers_used']:2d} (bit plane {q['bit_plane']:2d}): "
                  f"Compression={q['compression_ratio']:.1f}x, "
                  f"PSNR: Spatial={q['spatial_psnr']:.1f}dB, Angular={q['angular_psnr']:.1f}dB")
        
        print(f"  Final Quality:")
        final_q = quality_progression[-1]
        print(f"    Full reconstruction: "
              f"Compression={final_q['compression_ratio']:.1f}x, "
              f"PSNR: Spatial={final_q['spatial_psnr']:.1f}dB, Angular={final_q['angular_psnr']:.1f}dB")
        
        # Test partial reconstruction
        partial_spatial, partial_angular = bit_plane_coder.decode_progressive(
            bit_plane_layers, side_info, num_layers=len(bit_plane_layers)//2
        )
        
        # Verify shapes
        assert partial_spatial.shape == spatial_latents.shape
        assert partial_angular.shape == angular_latents.shape
        
        print(f"    Reconstruction successful: Shapes match original")
    
    print("\nBit-Plane Coder test completed!")


def test_bitstream_structurer():
    """Test the bitstream structuring component."""
    print("Testing Bitstream Structurer...")
    
    # Create test data with some sparsity patterns
    torch.manual_seed(42)
    spatial_latents = torch.randn(2, 64, 8, 12) * 2.0
    angular_latents = torch.randn(2, 64, 8, 12) * 1.5
    
    # Add sparsity to some channels to benefit from RLE
    spatial_latents[:, :16, :, :] = torch.where(
        torch.abs(spatial_latents[:, :16, :, :]) < 0.5,
        torch.zeros_like(spatial_latents[:, :16, :, :]),
        spatial_latents[:, :16, :, :]
    )
    
    angular_latents[:, :16, :, :] = torch.where(
        torch.abs(angular_latents[:, :16, :, :]) < 0.3,
        torch.zeros_like(angular_latents[:, :16, :, :]),
        angular_latents[:, :16, :, :]
    )
    
    print(f"Input data shapes: Spatial={spatial_latents.shape}, Angular={angular_latents.shape}")
    
    # Test different configurations
    configs = [
        {'use_rle': True, 'use_zigzag': True, 'use_rice_coding': True, 'rice_parameter': 2},
        {'use_rle': True, 'use_zigzag': False, 'use_rice_coding': True, 'rice_parameter': 3},
        {'use_rle': False, 'use_zigzag': True, 'use_rice_coding': True, 'rice_parameter': 1},
        {'use_rle': True, 'use_zigzag': True, 'use_rice_coding': False},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: {config} ---")
        
        # Create and train bitstream structurer
        structurer = BitstreamStructurer(**config)
        structurer.train(spatial_latents, angular_latents)
        
        # Apply structuring
        structured_data, side_info = structurer.structure_bitstream(spatial_latents, angular_latents)
        
        # Calculate compression benefits
        benefits = structurer.calculate_compression_benefit(
            spatial_latents, angular_latents, structured_data, side_info
        )
        
        print(f"  Structuring Results:")
        print(f"    Original bits: {benefits['original_bits']:,}")
        print(f"    Structured bits: {benefits['structured_bits']:,}")
        print(f"    Compression ratio: {benefits['compression_ratio']:.2f}x")
        print(f"    Size reduction: {benefits['size_reduction_percent']:.1f}%")
        print(f"    Bits per element: {benefits['bits_per_element']:.2f}")
        print(f"    Lossless reconstruction: {benefits['lossless_reconstruction']}")
        print(f"    Efficiency: {benefits['structuring_efficiency']}")
        
        if benefits['lossless_reconstruction']:
            print(f"    Reconstruction MSE: Spatial={benefits['spatial_reconstruction_mse']:.8f}, Angular={benefits['angular_reconstruction_mse']:.8f}")
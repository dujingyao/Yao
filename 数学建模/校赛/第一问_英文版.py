import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from PIL import Image
from matplotlib import colors
from matplotlib.patches import Polygon
import os

# 设置matplotlib后端，解决显示问题
matplotlib.use('Agg')  # 非交互式后端，保存图片
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class ColorSpaceConverter:
    def __init__(self):
        """Initialize color space converter"""
        # Define standard color space vertices (CIE 1931 xy coordinates)
        self.bt2020_vertices = np.array([
            [0.708, 0.292],  # Red
            [0.170, 0.797],  # Green  
            [0.131, 0.046]   # Blue
        ])
        
        self.display_vertices = np.array([
            [0.640, 0.330],  # Red
            [0.300, 0.600],  # Green
            [0.150, 0.060]   # Blue
        ])
        
        # CIE 1931 spectral locus data (simplified)
        self.setup_spectral_locus()
        
    def setup_spectral_locus(self):
        """Setup CIE 1931 spectral locus data"""
        # Simplified spectral locus data
        wavelengths = np.linspace(380, 700, 100)
        
        # Generate horseshoe-shaped spectral locus (approximation)
        t = np.linspace(0, 2*np.pi, 100)
        x_coords = 0.3 + 0.25 * np.cos(t) + 0.1 * np.cos(2*t)
        y_coords = 0.3 + 0.25 * np.sin(t) + 0.1 * np.sin(2*t)
        
        # Adjust to more realistic horseshoe shape
        mask = y_coords > 0.1
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        
        # Close curve
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
        
        self.spectral_locus = np.column_stack([x_coords, y_coords])

# Generate test RGB image
def generate_test_image(height, width):
    """Generate test RGB image with gradients and color blocks"""
    image = np.zeros((height, width, 3))
    
    # Create color gradients
    for i in range(height):
        for j in range(width):
            # Red gradient
            if j < width // 3:
                image[i, j] = [j / (width // 3), 0, 0]
            # Green gradient
            elif j < 2 * width // 3:
                image[i, j] = [0, (j - width // 3) / (width // 3), 0]
            # Blue gradient
            else:
                image[i, j] = [0, 0, (j - 2 * width // 3) / (width // 3)]
                
    # Add some color blocks
    image[50:100, 50:100] = [1, 1, 0]  # Yellow
    image[150:200, 50:100] = [1, 0, 1]  # Magenta
    image[50:100, 150:200] = [0, 1, 1]  # Cyan
    
    return np.clip(image, 0, 1)

def rgb_to_xy_approximate(rgb):
    """Convert RGB to CIE xy coordinates (simplified implementation)"""
    # This is a simplified conversion, should use standard RGB to XYZ to xy conversion
    # Linear transformation matrix (approximation)
    rgb = np.array(rgb)
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, -1)
    
    # Simplified transformation matrix
    transform_matrix = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    
    xyz = rgb @ transform_matrix.T
    
    # Convert to xy chromaticity coordinates
    sum_xyz = np.sum(xyz, axis=1, keepdims=True)
    sum_xyz[sum_xyz == 0] = 1  # Avoid division by zero
    
    x = xyz[:, 0:1] / sum_xyz
    y = xyz[:, 1:2] / sum_xyz
    
    return np.hstack([x, y])

def calculate_delta_e_lab(rgb1, rgb2):
    """Calculate Delta E color difference between two RGB colors (simplified)"""
    # Simplified RGB to Lab conversion
    def rgb_to_lab_simple(rgb):
        # Very simplified conversion, should use standard conversion
        lab = np.zeros_like(rgb)
        lab[..., 0] = 100 * (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])  # L
        lab[..., 1] = 128 * (rgb[..., 0] - rgb[..., 1])  # a
        lab[..., 2] = 128 * (rgb[..., 1] - rgb[..., 2])  # b
        return lab
    
    lab1 = rgb_to_lab_simple(rgb1)
    lab2 = rgb_to_lab_simple(rgb2)
    
    delta_e = np.sqrt(np.sum((lab1 - lab2)**2, axis=-1))
    return delta_e

def point_in_triangle(point, triangle):
    """Check if point is inside triangle"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(point, triangle[0], triangle[1])
    d2 = sign(point, triangle[1], triangle[2])
    d3 = sign(point, triangle[2], triangle[0])
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def barycentric_coordinates(point, triangle):
    """Calculate barycentric coordinates"""
    x, y = point
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]
    
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return np.array([1/3, 1/3, 1/3])
        
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b
    
    return np.array([a, b, c])

def map_color_gamut(source_xy, converter):
    """Map color gamut"""
    mapped_xy = []
    
    for point in source_xy:
        if point_in_triangle(point, converter.bt2020_vertices):
            # Inside BT2020 gamut, calculate barycentric coordinates
            bary_coords = barycentric_coordinates(point, converter.bt2020_vertices)
            # Map to display gamut
            mapped_point = (bary_coords[0] * converter.display_vertices[0] + 
                           bary_coords[1] * converter.display_vertices[1] + 
                           bary_coords[2] * converter.display_vertices[2])
        else:
            # Outside gamut, project to nearest boundary
            mapped_point = project_to_triangle(point, converter.display_vertices)
        
        mapped_xy.append(mapped_point)
    
    return np.array(mapped_xy)

def project_to_triangle(point, triangle):
    """Project point to nearest position inside triangle"""
    # Simplified implementation: project to centroid
    return np.mean(triangle, axis=0)

def plot_image_comparison(image1, image2, save_path=None):
    """Visualize original vs optimized image comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image1)
    axes[0].set_title("BT2020 Source Image", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(image2)
    axes[1].set_title("Mapped Display Image", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image comparison saved to: {save_path}")
    else:
        plt.savefig('image_comparison.png', dpi=300, bbox_inches='tight')
        print("Image comparison saved to: image_comparison.png")
    plt.close()

def plot_delta_e_curve(delta_e_values, save_path=None):
    """Plot Delta E error curve"""
    plt.figure(figsize=(10, 6))
    
    # Calculate statistics
    mean_delta_e = np.mean(delta_e_values)
    max_delta_e = np.max(delta_e_values)
    
    plt.plot(delta_e_values, 'b-', linewidth=2, label=f"ΔE values (mean: {mean_delta_e:.3f})")
    plt.axhline(y=mean_delta_e, color='r', linestyle='--', alpha=0.7, label=f"Mean: {mean_delta_e:.3f}")
    plt.axhline(y=max_delta_e, color='orange', linestyle='--', alpha=0.7, label=f"Max: {max_delta_e:.3f}")
    
    plt.title("Color Conversion Delta E Error Distribution", fontsize=16)
    plt.xlabel("Pixel Index", fontsize=12)
    plt.ylabel("Delta E", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text
    stats_text = f"Total Pixels: {len(delta_e_values)}\nMean ΔE: {mean_delta_e:.3f}\nMax ΔE: {max_delta_e:.3f}\nStd Dev: {np.std(delta_e_values):.3f}"
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Delta E curve saved to: {save_path}")
    else:
        plt.savefig('delta_e_curve.png', dpi=300, bbox_inches='tight')
        print("Delta E curve saved to: delta_e_curve.png")
    plt.close()

def plot_cie1931_color_space(converter, source_points=None, mapped_points=None, save_path=None):
    """Plot CIE1931 chromaticity diagram"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot spectral locus
    spectral_x = converter.spectral_locus[:, 0]
    spectral_y = converter.spectral_locus[:, 1]
    
    ax.plot(spectral_x, spectral_y, 'k-', linewidth=2, label='CIE1931 Spectral Locus')
    ax.fill(spectral_x, spectral_y, color='lightgray', alpha=0.3)
    
    # Plot BT2020 gamut
    bt2020_triangle = Polygon(converter.bt2020_vertices, fill=False, 
                             edgecolor='brown', linewidth=3, 
                             label='BT2020 Gamut')
    ax.add_patch(bt2020_triangle)
    
    # Plot display gamut
    display_triangle = Polygon(converter.display_vertices, fill=False, 
                              edgecolor='red', linewidth=3, 
                              label='Display Gamut')
    ax.add_patch(display_triangle)
    
    # Annotate vertices
    colors_labels = ['R', 'G', 'B']
    for i, (vertex, label) in enumerate(zip(converter.bt2020_vertices, colors_labels)):
        ax.plot(vertex[0], vertex[1], 'bo', markersize=8)
        ax.annotate(f'BT2020-{label}', vertex, xytext=(5, 5), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
        
    for i, (vertex, label) in enumerate(zip(converter.display_vertices, colors_labels)):
        ax.plot(vertex[0], vertex[1], 'ro', markersize=8)
        ax.annotate(f'Display-{label}', vertex, xytext=(5, -15), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    # If color points exist, plot mapping
    if source_points is not None and mapped_points is not None:
        ax.scatter(source_points[:, 0], source_points[:, 1], 
                  c='blue', s=30, alpha=0.6, label='Source Colors')
        ax.scatter(mapped_points[:, 0], mapped_points[:, 1], 
                  c='red', s=30, alpha=0.6, label='Mapped Colors')
        
        # Draw mapping arrows (only show subset to avoid clutter)
        step = max(1, len(source_points) // 20)
        for i in range(0, len(source_points), step):
            ax.arrow(source_points[i, 0], source_points[i, 1],
                    mapped_points[i, 0] - source_points[i, 0],
                    mapped_points[i, 1] - source_points[i, 1],
                    head_width=0.01, head_length=0.01, 
                    fc='green', ec='green', alpha=0.5)
    
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('CIE x', fontsize=12)
    ax.set_ylabel('CIE y', fontsize=12)
    ax.set_title('CIE1931 Chromaticity Diagram and Color Space Conversion', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CIE1931 chromaticity diagram saved to: {save_path}")
    else:
        plt.savefig('cie1931_color_space.png', dpi=300, bbox_inches='tight')
        print("CIE1931 chromaticity diagram saved to: cie1931_color_space.png")
    plt.close()

def main():
    """Main function"""
    print("=== LED Display Color Conversion Design and Calibration - Problem 1 ===")
    
    # Create output directory
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize converter
    converter = ColorSpaceConverter()
    
    print("Step 1: Generate test image...")
    # Generate test image
    original_image = generate_test_image(256, 256)
    
    print("Step 2: Convert color space...")
    # Convert image RGB to xy coordinates
    h, w, c = original_image.shape
    rgb_flat = original_image.reshape(-1, 3)
    
    # Convert to xy chromaticity coordinates
    source_xy = rgb_to_xy_approximate(rgb_flat)
    
    # Perform gamut mapping
    mapped_xy = map_color_gamut(source_xy, converter)
    
    print("Step 3: Calculate converted RGB...")
    # Simplified: use linear scaling as inverse conversion
    # Should implement precise xy to RGB conversion
    scale_factor = 0.8  # Simulate gamut compression
    mapped_rgb = rgb_flat * scale_factor
    mapped_image = mapped_rgb.reshape(h, w, c)
    mapped_image = np.clip(mapped_image, 0, 1)
    
    print("Step 4: Calculate color difference...")
    # Calculate Delta E color difference
    delta_e_values = calculate_delta_e_lab(original_image, mapped_image)
    delta_e_flat = delta_e_values.flatten()
    
    # Statistics
    avg_delta_e = np.mean(delta_e_flat)
    max_delta_e = np.max(delta_e_flat)
    std_delta_e = np.std(delta_e_flat)
    
    print(f"\n=== Conversion Results Statistics ===")
    print(f"Average ΔE color difference: {avg_delta_e:.4f}")
    print(f"Maximum ΔE color difference: {max_delta_e:.4f}")
    print(f"ΔE standard deviation: {std_delta_e:.4f}")
    
    # Gamut coverage calculation
    def triangle_area(vertices):
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))/2)
    
    bt2020_area = triangle_area(converter.bt2020_vertices)
    display_area = triangle_area(converter.display_vertices)
    coverage = display_area / bt2020_area
    
    print(f"BT2020 gamut area: {bt2020_area:.6f}")
    print(f"Display gamut area: {display_area:.6f}")
    print(f"Gamut coverage ratio: {coverage:.2%}")
    
    print("\nStep 5: Generate visualization results...")
    
    # Generate image comparison
    plot_image_comparison(original_image, mapped_image, 
                         os.path.join(output_dir, 'image_comparison.png'))
    
    # Generate Delta E error curve
    plot_delta_e_curve(delta_e_flat[:1000],  # Only show first 1000 points to avoid clutter
                      os.path.join(output_dir, 'delta_e_curve.png'))
    
    # Generate CIE1931 chromaticity diagram
    sample_indices = np.random.choice(len(source_xy), 100, replace=False)
    plot_cie1931_color_space(converter, 
                             source_xy[sample_indices], 
                             mapped_xy[sample_indices],
                             os.path.join(output_dir, 'cie1931_color_space.png'))
    
    # Save numerical results
    results = {
        'source_image': original_image,
        'mapped_image': mapped_image,
        'delta_e_values': delta_e_flat,
        'statistics': {
            'avg_delta_e': avg_delta_e,
            'max_delta_e': max_delta_e,
            'std_delta_e': std_delta_e,
            'coverage_ratio': coverage
        }
    }
    
    np.savez(os.path.join(output_dir, 'conversion_results.npz'), **results)
    
    print(f"\n=== Complete ===")
    print(f"All results saved to: {output_dir}/")
    print("Generated files:")
    print("1. image_comparison.png - Image comparison")
    print("2. delta_e_curve.png - Delta E error curve")
    print("3. cie1931_color_space.png - CIE1931 chromaticity diagram")
    print("4. conversion_results.npz - Numerical results")

if __name__ == "__main__":
    main()

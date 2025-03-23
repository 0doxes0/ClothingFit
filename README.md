# Clothing Fit Tool for Blender

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![cn](https://img.shields.io/badge/语言-中文-red.svg)](README.zh-CN.md)
[![jp](https://img.shields.io/badge/言語-日本語-white.svg)](README.ja.md)

A simple Blender add-on designed to accelerate the process of fitting clothing to 3D character models. This tool helps create a snug fit for garments that are already close to a character's shape, while preserving overall form and wrinkle details. If you're struggling with clothing fit and haven't used specialized tools before, this might be worth trying.

## Features

The tool adjusts clothing mesh vertices to an ideal distance from the character model's surface using various projection methods:

- Multiple projection types: from center point, vertex normals, nearest point, and mixed methods
- Adjustable influence range and falloff functions
- Automatic detection and fixing of intersecting vertices
- Interactive vertex selection
- Support for fine adjustments in specific areas

## Installation

1. Download the `clothing_fit_tool.py` file
2. In Blender, navigate to `Edit > Preferences > Add-ons > Install...`
3. Select the downloaded py file
4. Enable the add-on

## Usage Guide

### Basic Workflow

1. Find the "Clothing Fit Tool" panel in the 3D View sidebar
2. Select the clothing object and character body object
3. Adjust parameters (soft range, falloff type, ideal distance, etc.)
4. Enter Edit Mode and select vertices to fit (Due to N² complexity, select reference points rather than all vertices - e.g., garment edges - nearby points will follow based on Soft Range)
5. Click "Apply Clothing Fit" to apply the adjustments

### Parameter Explanation

- **Clothing Object**: The garment mesh to be fitted
- **Body Object**: The character body mesh
- **Soft Range**: Influence radius determining how far selected vertices will affect surrounding geometry
- **Falloff Type**: Type of influence falloff (linear, quadratic, exponential, etc. - cosine recommended for smoothest results)
- **Ideal Distance**: Target distance between clothing and body
- **Projection Type**: Method to calculate projection
  - **From Center**: Projects from center point (may cause collapse)
  - **Vertex Normal**: Projects along vertex normals (automatically filters poor normals)
  - **Nearest Point**: Finds nearest point on body mesh
  - **Mixed Method**: Combines multiple projection methods (may cause collapse)
- **Ray Offset**: Distance to offset ray start position along normal direction

### Tips & Tricks

1. **Multi-stage Fitting**: 
   - Start with larger Soft Range for overall adjustments
   - Gradually reduce Soft Range for fine-tuning

2. **Choosing Projection Methods**:
   - "Mixed Method" works best for most cases
   - Use "Vertex Normal" for complex areas like armpits and crotch

3. **Handling Intersections**:
   - Intersections are common - use "Fix Intersecting" to resolve them
   - For manual fixes, use "Select Intersecting" to identify problematic vertices in Edit Mode

4. **Selective Adjustment**:
   - For best performance and control, select few key reference points rather than many vertices
   - Denser vertex selections will be considered multiple times - ensure reference vertices are evenly distributed

## Limitations

- Current implementation is basic and primarily helps speed up mesh intersection and fitting workflows
- Weight propagation mechanism is slow and not suitable for thousands of vertices at once
- Some special cases may still require manual adjustments

## Potential Improvements

- Improving deformation with Laplacian interpolation
- Optimizing weight calculations for better performance
- Developing more intelligent intersection detection and fixing algorithms

## Additional Notes

- This tool, including this README, was created with AI assistance
- This is a temporary solution to accelerate clothing fitting. Looking forward to the completion of automatic clothing fitting tools for Unity being developed by talented individuals.

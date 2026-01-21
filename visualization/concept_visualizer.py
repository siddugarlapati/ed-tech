#!/usr/bin/env python3
"""
Advanced 3D Concept Visualizer
Generates interactive 3D visualizations using GSAP and Anime.js
Dynamically creates visualizations based on educational concepts
"""

import json
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass


class ConceptType(Enum):
    """Types of educational concepts that can be visualized."""
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    COMPUTER_SCIENCE = "computer_science"
    GEOMETRY = "geometry"


class AnimationLibrary(Enum):
    """Animation libraries available for visualizations."""
    GSAP = "gsap"
    ANIME_JS = "anime"
    THREE_JS = "three"


@dataclass
class VisualizationConfig:
    """Configuration for 3D visualization generation."""
    concept_type: ConceptType
    difficulty_level: int  # 1-10
    animation_library: AnimationLibrary
    interactive: bool = True
    show_labels: bool = True
    show_equations: bool = True
    animation_speed: float = 1.0


class ConceptVisualizer:
    """
    Generates interactive 3D visualizations for educational concepts.
    Uses GSAP, Anime.js, and Three.js for smooth animations.
    """
    
    def __init__(self):
        self.concept_templates = {
            "unit_circle": self._generate_unit_circle,
            "sine_wave": self._generate_sine_wave,
            "pythagorean_theorem": self._generate_pythagorean,
            "vector_addition": self._generate_vector_addition,
            "function_graph": self._generate_function_graph,
        }
    
    def generate_visualization(
        self,
        concept_name: str,
        config: VisualizationConfig,
        **params
    ) -> str:
        """Generate HTML/JS visualization for a concept."""
        
        if concept_name in self.concept_templates:
            generator = self.concept_templates[concept_name]
            return generator(config, **params)
        else:
            return self._generate_default_visualization(concept_name, config)
    
    def _generate_unit_circle(
        self,
        config: VisualizationConfig,
        angle: float = 45,
        **kwargs
    ) -> str:
        """Generate interactive unit circle with GSAP animations."""
        
        # Build HTML template without f-string to avoid double braces
        html_parts = []
        
        html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Unit Circle</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow: hidden;
        }
        #canvas-container { 
            width: 100vw; 
            height: 100vh; 
            position: relative;
        }
        #controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            z-index: 100;
            min-width: 300px;
        }
        #info {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            z-index: 100;
            min-width: 250px;
        }
        h3 { 
            margin: 0 0 15px 0; 
            color: #667eea;
            font-size: 18px;
        }
        .control-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #333;
            font-weight: 500;
            font-size: 14px;
        }
        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
            background: #764ba2;
            transform: scale(1.2);
        }
        button {
            width: 100%;
            padding: 12px;
            margin: 5px 0;
            border: none;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        .value-display {
            display: inline-block;
            background: #f0f0f0;
            padding: 4px 10px;
            border-radius: 5px;
            font-weight: 600;
            color: #667eea;
            margin-left: 10px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        .info-label {
            font-weight: 600;
            color: #555;
        }
        .info-value {
            color: #667eea;
            font-weight: 700;
        }
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    
    <div id="controls">
        <h3>🎛️ Controls</h3>
        <div class="control-group">
            <label>
                Angle: <span class="value-display" id="angleDisplay">''')
        
        html_parts.append(str(angle))
        html_parts.append('''°</span>
            </label>
            <input type="range" id="angleSlider" min="0" max="360" value="''')
        html_parts.append(str(angle))
        html_parts.append('''" step="1">
        </div>
        <div class="control-group">
            <label>
                Animation Speed: <span class="value-display" id="speedDisplay">''')
        html_parts.append(str(config.animation_speed))
        html_parts.append('''x</span>
            </label>
            <input type="range" id="speedSlider" min="0.1" max="3" value="''')
        html_parts.append(str(config.animation_speed))
        html_parts.append('''" step="0.1">
        </div>
        <button id="playBtn">▶️ Play Animation</button>
        <button id="resetBtn">🔄 Reset View</button>
        <button id="toggleProjections">📐 Toggle Projections</button>
    </div>
    
    <div id="info">
        <h3>📊 Trigonometric Values</h3>
        <div class="info-row">
            <span class="info-label">sin(θ)</span>
            <span class="info-value" id="sinValue">0.707</span>
        </div>
        <div class="info-row">
            <span class="info-label">cos(θ)</span>
            <span class="info-value" id="cosValue">0.707</span>
        </div>
        <div class="info-row">
            <span class="info-label">tan(θ)</span>
            <span class="info-value" id="tanValue">1.000</span>
        </div>
        <div class="info-row">
            <span class="info-label">Angle (rad)</span>
            <span class="info-value" id="radValue">0.785</span>
        </div>
    </div>

    <script>
        // Configuration
        const CONFIG = {
            initialAngle: ''')
        html_parts.append(str(angle))
        html_parts.append(''',
            animationSpeed: ''')
        html_parts.append(str(config.animation_speed))
        html_parts.append('''
        };
        
        // Three.js Scene Setup
        let scene, camera, renderer;
        let circle, anglePoint, radiusLine, projectionLines;
        let isAnimating = false;
        let currentAngle = CONFIG.initialAngle;
        let animationSpeed = CONFIG.animationSpeed;
        let showProjections = true;
        
        function init() {
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            // Camera
            camera = new THREE.PerspectiveCamera(
                75,
                window.innerWidth / window.innerHeight,
                0.1,
                1000
            );
            camera.position.set(0, 0, 5);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            document.getElementById('canvas-container').appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const pointLight = new THREE.PointLight(0xffffff, 0.8);
            pointLight.position.set(5, 5, 5);
            scene.add(pointLight);
            
            createUnitCircle();
            createAxes();
            createAngleVisualization();
            
            // Event Listeners
            document.getElementById('angleSlider').addEventListener('input', updateAngle);
            document.getElementById('speedSlider').addEventListener('input', updateSpeed);
            document.getElementById('playBtn').addEventListener('click', toggleAnimation);
            document.getElementById('resetBtn').addEventListener('click', resetView);
            document.getElementById('toggleProjections').addEventListener('click', toggleProjectionsView);
            window.addEventListener('resize', onWindowResize);
            
            animate();
        }
        
        function createUnitCircle() {
            // Circle outline
            const circleGeometry = new THREE.RingGeometry(0.98, 1.02, 128);
            const circleMaterial = new THREE.MeshBasicMaterial({
                color: 0x00ff88,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.8
            });
            circle = new THREE.Mesh(circleGeometry, circleMaterial);
            scene.add(circle);
            
            // Animate circle appearance with GSAP
            gsap.from(circle.scale, {
                x: 0,
                y: 0,
                z: 0,
                duration: 1.5,
                ease: "elastic.out(1, 0.5)"
            });
        }
        
        function createAxes() {
            const axisLength = 2.5;
            const axisWidth = 0.02;
            
            // X-axis (red)
            const xGeometry = new THREE.CylinderGeometry(axisWidth, axisWidth, axisLength, 16);
            const xMaterial = new THREE.MeshLambertMaterial({ color: 0xff0000 });
            const xAxis = new THREE.Mesh(xGeometry, xMaterial);
            xAxis.rotation.z = Math.PI / 2;
            scene.add(xAxis);
            
            // Y-axis (green)
            const yGeometry = new THREE.CylinderGeometry(axisWidth, axisWidth, axisLength, 16);
            const yMaterial = new THREE.MeshLambertMaterial({ color: 0x00ff00 });
            const yAxis = new THREE.Mesh(yGeometry, yMaterial);
            scene.add(yAxis);
            
            // Animate axes
            gsap.from([xAxis.scale, yAxis.scale], {
                y: 0,
                duration: 1,
                ease: "power2.out",
                stagger: 0.2
            });
        }
        
        function createAngleVisualization() {
            // Point on circle
            const pointGeometry = new THREE.SphereGeometry(0.08, 32, 32);
            const pointMaterial = new THREE.MeshLambertMaterial({ 
                color: 0xffff00,
                emissive: 0xffff00,
                emissiveIntensity: 0.5
            });
            anglePoint = new THREE.Mesh(pointGeometry, pointMaterial);
            scene.add(anglePoint);
            
            // Radius line
            const lineGeometry = new THREE.BufferGeometry();
            const lineMaterial = new THREE.LineBasicMaterial({ 
                color: 0xffff00,
                linewidth: 3
            });
            radiusLine = new THREE.Line(lineGeometry, lineMaterial);
            scene.add(radiusLine);
            
            // Projection lines group
            projectionLines = new THREE.Group();
            scene.add(projectionLines);
            
            updateVisualization();
        }
        
        function updateVisualization() {
            const angleRad = (currentAngle * Math.PI) / 180;
            const x = Math.cos(angleRad);
            const y = Math.sin(angleRad);
            
            // Update point position with GSAP animation
            gsap.to(anglePoint.position, {
                x: x,
                y: y,
                z: 0,
                duration: 0.3,
                ease: "power2.out"
            });
            
            // Update radius line
            const positions = new Float32Array([
                0, 0, 0,
                x, y, 0
            ]);
            radiusLine.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            // Update projection lines
            projectionLines.clear();
            
            if (showProjections) {
                // Cosine projection (to x-axis)
                const cosLineGeometry = new THREE.BufferGeometry();
                cosLineGeometry.setAttribute('position', new THREE.BufferAttribute(
                    new Float32Array([x, y, 0, x, 0, 0]), 3
                ));
                const cosLine = new THREE.Line(
                    cosLineGeometry,
                    new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2 })
                );
                projectionLines.add(cosLine);
                
                // Sine projection (to y-axis)
                const sinLineGeometry = new THREE.BufferGeometry();
                sinLineGeometry.setAttribute('position', new THREE.BufferAttribute(
                    new Float32Array([x, y, 0, 0, y, 0]), 3
                ));
                const sinLine = new THREE.Line(
                    sinLineGeometry,
                    new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 })
                );
                projectionLines.add(sinLine);
            }
            
            // Update display values
            document.getElementById('sinValue').textContent = y.toFixed(3);
            document.getElementById('cosValue').textContent = x.toFixed(3);
            document.getElementById('tanValue').textContent = (y / x).toFixed(3);
            document.getElementById('radValue').textContent = angleRad.toFixed(3);
        }
        
        function updateAngle() {
            currentAngle = parseFloat(document.getElementById('angleSlider').value);
            document.getElementById('angleDisplay').textContent = currentAngle.toFixed(0) + '°';
            updateVisualization();
        }
        
        function updateSpeed() {
            animationSpeed = parseFloat(document.getElementById('speedSlider').value);
            document.getElementById('speedDisplay').textContent = animationSpeed.toFixed(1) + 'x';
        }
        
        function toggleAnimation() {
            isAnimating = !isAnimating;
            document.getElementById('playBtn').textContent = isAnimating ? '⏸️ Pause' : '▶️ Play Animation';
        }
        
        function resetView() {
            currentAngle = 45;
            document.getElementById('angleSlider').value = 45;
            updateAngle();
            
            gsap.to(camera.position, {
                x: 0,
                y: 0,
                z: 5,
                duration: 1,
                ease: "power2.inOut"
            });
        }
        
        function toggleProjectionsView() {
            showProjections = !showProjections;
            updateVisualization();
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            if (isAnimating) {
                currentAngle += animationSpeed;
                if (currentAngle >= 360) currentAngle = 0;
                document.getElementById('angleSlider').value = currentAngle;
                updateAngle();
            }
            
            scene.rotation.z += 0.001;
            renderer.render(scene, camera);
        }
        
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
        
        // Initialize
        init();
    </script>
</body>
</html>''')
        
        return ''.join(html_parts)
    
    def _generate_sine_wave(self, config: VisualizationConfig, **kwargs) -> str:
        """Generate animated sine wave."""
        return self._generate_default_visualization("sine_wave", config)
    
    def _generate_pythagorean(self, config: VisualizationConfig, **kwargs) -> str:
        """Generate Pythagorean theorem visualization."""
        return self._generate_default_visualization("pythagorean", config)
    
    def _generate_vector_addition(self, config: VisualizationConfig, **kwargs) -> str:
        """Generate vector addition visualization."""
        return self._generate_default_visualization("vectors", config)
    
    def _generate_function_graph(self, config: VisualizationConfig, **kwargs) -> str:
        """Generate function graphing visualization."""
        return self._generate_default_visualization("function", config)
    
    def _generate_default_visualization(self, concept_name: str, config: VisualizationConfig) -> str:
        """Generate a default placeholder visualization."""
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{concept_name} Visualization</title>
    <style>
        body {{
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: Arial, sans-serif;
        }}
        .message {{
            text-align: center;
            color: white;
            font-size: 24px;
        }}
    </style>
</head>
<body>
    <div class="message">
        <h1>🎨 {concept_name.replace('_', ' ').title()}</h1>
        <p>Visualization coming soon!</p>
        <p>Difficulty Level: {config.difficulty_level}/10</p>
    </div>
</body>
</html>'''


def main():
    """Demo of the visualization system."""
    visualizer = ConceptVisualizer()
    
    config = VisualizationConfig(
        concept_type=ConceptType.MATHEMATICS,
        difficulty_level=5,
        animation_library=AnimationLibrary.GSAP,
        interactive=True,
        show_labels=True,
        show_equations=True,
        animation_speed=1.0
    )
    
    # Generate unit circle visualization
    html = visualizer.generate_visualization("unit_circle", config, angle=45)
    
    # Save to file
    with open("unit_circle_demo.html", "w") as f:
        f.write(html)
    
    print("✅ Visualization generated: unit_circle_demo.html")
    print("🌐 Open in browser to view interactive 3D visualization")


if __name__ == "__main__":
    main()

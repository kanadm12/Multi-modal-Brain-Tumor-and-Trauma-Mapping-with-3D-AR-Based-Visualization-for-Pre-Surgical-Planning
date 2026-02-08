'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Box, CircularProgress, Typography } from '@mui/material';
import { MeshResponse } from '@/services/api';

interface BrainViewerProps {
  meshData: MeshResponse | null;
  loading?: boolean;
  showBrain?: boolean;
  showNCR?: boolean;
  showED?: boolean;
  showET?: boolean;
  brainOpacity?: number;
  scale?: number;
}

// Tumor class colors
const TUMOR_COLORS = {
  NCR: 0x8B0000, // Dark Red
  ED: 0xFFD700,  // Yellow  
  ET: 0xFF0000,  // Bright Red
};

const BrainViewer: React.FC<BrainViewerProps> = ({
  meshData,
  loading = false,
  showBrain = true,
  showNCR = true,
  showED = true,
  showET = true,
  brainOpacity = 0.15,
  scale = 1,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const animationIdRef = useRef<number | null>(null);

  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize Three.js scene
  const initScene = useCallback(() => {
    if (!containerRef.current || isInitialized) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    sceneRef.current = scene;

    // Create camera
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 2000);
    camera.position.set(0, 0, 300);
    cameraRef.current = camera;

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true 
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableZoom = true;
    controls.enablePan = true;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 0.5;
    controlsRef.current = controls;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(100, 100, 100);
    directionalLight1.castShadow = true;
    scene.add(directionalLight1);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-100, -100, -100);
    scene.add(directionalLight2);

    // Add hemisphere light for better ambient
    const hemiLight = new THREE.HemisphereLight(0xddeeff, 0x0f0e0d, 0.4);
    scene.add(hemiLight);

    // Create mesh group
    const meshGroup = new THREE.Group();
    meshGroup.name = 'brainMeshGroup';
    scene.add(meshGroup);
    meshGroupRef.current = meshGroup;

    // Add grid helper (optional, for debugging)
    // const gridHelper = new THREE.GridHelper(200, 20);
    // gridHelper.rotation.x = Math.PI / 2;
    // scene.add(gridHelper);

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    setIsInitialized(true);

    // Handle resize
    const handleResize = () => {
      if (!container || !camera || !renderer) return;
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [isInitialized]);

  // Create mesh from data
  const createMesh = useCallback((
    meshData: { vertices: number[][]; faces: number[][] },
    color: number,
    opacity: number,
    name: string
  ): THREE.Mesh => {
    const geometry = new THREE.BufferGeometry();

    // Flatten vertices array
    const vertices = new Float32Array(meshData.vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    // Create indices from faces
    const indices = new Uint32Array(meshData.faces.flat());
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

    // Compute normals for proper lighting
    geometry.computeVertexNormals();

    // Center the geometry
    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox?.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);

    // Create material
    const material = new THREE.MeshPhongMaterial({
      color: color,
      transparent: opacity < 1,
      opacity: opacity,
      side: THREE.DoubleSide,
      flatShading: false,
      shininess: 30,
      depthWrite: opacity >= 0.5,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = name;
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    return mesh;
  }, []);

  // Update meshes when data changes
  useEffect(() => {
    if (!meshGroupRef.current || !meshData) return;

    const group = meshGroupRef.current;

    // Clear existing meshes
    while (group.children.length > 0) {
      const child = group.children[0];
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        if (child.material instanceof THREE.Material) {
          child.material.dispose();
        }
      }
      group.remove(child);
    }

    // Add brain mesh
    if (meshData.brain_mesh && meshData.brain_mesh.vertices.length > 0) {
      const brainMesh = createMesh(
        meshData.brain_mesh,
        0xE5E5F2,
        brainOpacity,
        'brain'
      );
      brainMesh.visible = showBrain;
      group.add(brainMesh);
    }

    // Add tumor meshes
    const tumorKeys = ['NCR', 'ED', 'ET'] as const;
    const visibilityMap = { NCR: showNCR, ED: showED, ET: showET };

    for (const key of tumorKeys) {
      const tumorData = meshData.tumor_meshes[key];
      if (tumorData && tumorData.vertices.length > 0) {
        const color = TUMOR_COLORS[key];
        const opacity = key === 'ED' ? 0.7 : 1.0;
        const tumorMesh = createMesh(tumorData, color, opacity, key);
        tumorMesh.visible = visibilityMap[key];
        group.add(tumorMesh);
      }
    }

    // Auto-fit camera to mesh bounds
    if (group.children.length > 0 && cameraRef.current && controlsRef.current) {
      const box = new THREE.Box3().setFromObject(group);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const camera = cameraRef.current;
      const controls = controlsRef.current;
      
      camera.position.set(0, 0, maxDim * 2);
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }, [meshData, brainOpacity, createMesh, showBrain, showNCR, showED, showET]);

  // Update visibility when toggles change
  useEffect(() => {
    if (!meshGroupRef.current) return;

    meshGroupRef.current.children.forEach((child: THREE.Object3D) => {
      if (child instanceof THREE.Mesh) {
        switch (child.name) {
          case 'brain':
            child.visible = showBrain;
            if (child.material instanceof THREE.MeshPhongMaterial) {
              child.material.opacity = brainOpacity;
            }
            break;
          case 'NCR':
            child.visible = showNCR;
            break;
          case 'ED':
            child.visible = showED;
            break;
          case 'ET':
            child.visible = showET;
            break;
        }
      }
    });
  }, [showBrain, showNCR, showED, showET, brainOpacity]);

  // Update scale
  useEffect(() => {
    if (meshGroupRef.current) {
      meshGroupRef.current.scale.setScalar(scale);
    }
  }, [scale]);

  // Initialize scene on mount
  useEffect(() => {
    initScene();
    const container = containerRef.current;

    return () => {
      // Cleanup
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (rendererRef.current && container) {
        container.removeChild(rendererRef.current.domElement);
        rendererRef.current.dispose();
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        height: '100%',
        position: 'relative',
        borderRadius: 2,
        overflow: 'hidden',
      }}
    >
      {loading && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(26, 26, 46, 0.8)',
            zIndex: 10,
          }}
        >
          <CircularProgress size={60} />
          <Typography sx={{ mt: 2, color: 'white' }}>
            Loading 3D model...
          </Typography>
        </Box>
      )}

      {!loading && !meshData && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#1a1a2e',
          }}
        >
          <Typography color="text.secondary">
            Upload MRI scans to view 3D model
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default BrainViewer;

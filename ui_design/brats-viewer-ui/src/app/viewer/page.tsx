"use client";

import React, { useState, useEffect, Suspense } from 'react';
import { 
    Box, 
    Typography, 
    Container, 
    Paper, 
    Slider, 
    Checkbox, 
    FormControlLabel,
    Button,
    Divider,
    Chip,
    CircularProgress
} from '@mui/material';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import DescriptionIcon from '@mui/icons-material/Description';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useSearchParams } from 'next/navigation';
import dynamic from 'next/dynamic';
import { apiService, MeshResponse, ReportResponse } from '@/services/api';
import { STLManifest } from '@/components/BrainViewer';

// Dynamically import BrainViewer to avoid SSR issues with Three.js
const BrainViewer = dynamic(() => import('@/components/BrainViewer'), { 
    ssr: false,
    loading: () => (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <CircularProgress />
        </Box>
    )
});

// Demo report data - cast to unknown first to bypass strict type checking
const DEMO_REPORT = {
    report_id: 'demo-report-001',
    session_id: 'demo-session',
    generated_at: new Date().toISOString(),
    hospital_name: 'Demo Hospital',
    patient: {
        id: 'demo-patient-001',
        name: 'Demo Patient',
        age: '45',
        gender: 'Unknown'
    },
    patient_info: {
        name: 'Demo Patient',
        age: '45',
        gender: 'Unknown'
    },
    doctor: {
        name: 'Dr. Demo',
        department: 'Neurology',
        credentials: 'MD, PhD'
    },
    tumor_analysis: {
        classes: {},
        whole_tumor_volume_cm3: 42.5,
        tumor_core_volume_cm3: 18.3,
        enhancing_tumor_volume_cm3: 12.1,
        estimated_location: {
            region: 'Frontal Lobe',
            hemisphere: 'Right',
            coordinates: { axial_slice: 45, coronal_slice: 67, sagittal_slice: 89 }
        },
        estimated_grade: {
            grade: 'High Grade Glioma (HGG)',
            confidence: '94.2%',
            description: 'Aggressive tumor requiring immediate attention'
        }
    },
    clinical_findings: [
        'Tumor detected in right frontal lobe',
        'Estimated volume: 42.5 cm³',
        'Well-defined margins observed'
    ],
    recommendations: [
        'Consult with neuro-oncologist for treatment planning',
        'Consider surgical resection evaluation',
        'Follow-up MRI recommended in 2-4 weeks'
    ],
    disclaimer: 'This is a demo report for demonstration purposes only.'
} as unknown as ReportResponse;

function ViewerContent() {
    const searchParams = useSearchParams();
    const sessionId = searchParams.get('session');
    const isDemo = searchParams.get('demo') === 'true';

    const [meshData, setMeshData] = useState<MeshResponse | null>(null);
    const [report, setReport] = useState<ReportResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [glbUrl, setGlbUrl] = useState<string | null>(null);
    const [stlManifest, setStlManifest] = useState<STLManifest | null>(null);

    // Display controls
    const [showBrain, setShowBrain] = useState(true);
    const [showNCR, setShowNCR] = useState(true);
    const [showED, setShowED] = useState(true);
    const [showET, setShowET] = useState(true);
    const [brainOpacity, setBrainOpacity] = useState(15);
    const [scale, setScale] = useState(100);

    const loadDataForSession = async (sid: string) => {
        setLoading(true);
        setError(null);

        try {
            const [meshResult, reportResult] = await Promise.all([
                apiService.getMeshData(sid),
                apiService.getReport(sid)
            ]);

            setMeshData(meshResult);
            setReport(reportResult);
        } catch (err) {
            console.error('Failed to load data:', err);
            setError(err instanceof Error ? err.message : 'Failed to load visualization data');
        } finally {
            setLoading(false);
        }
    };

    const loadDemoData = async () => {
        setLoading(true);
        try {
            // Load STL manifest
            const response = await fetch('/demo_manifest.json');
            const manifest: STLManifest = await response.json();
            setStlManifest(manifest);
            
            // Create report from manifest stats
            const demoReport = {
                ...DEMO_REPORT,
                tumor_analysis: {
                    ...DEMO_REPORT.tumor_analysis,
                    whole_tumor_volume_cm3: manifest.total_tumor_volume_cm3 || 28.5,
                    tumor_core_volume_cm3: manifest.stats?.ET?.volume_cm3 || 3.9,
                    enhancing_tumor_volume_cm3: manifest.stats?.ET?.volume_cm3 || 3.9,
                }
            };
            setReport(demoReport);
        } catch (err) {
            console.error('Failed to load demo data:', err);
            // Fallback to GLB if manifest fails
            setGlbUrl('/demo_brain_tumor.glb');
            setReport(DEMO_REPORT);
        }
        
        // Try to get patient info from localStorage
        const patientInfo = localStorage.getItem('patientInfo');
        if (patientInfo) {
            try {
                const parsed = JSON.parse(patientInfo);
                setReport(prev => prev ? {
                    ...prev,
                    patient_info: {
                        name: parsed.name || 'Demo Patient',
                        age: parsed.age || '45',
                        gender: parsed.gender || 'Unknown'
                    }
                } : prev);
            } catch {
                // Use default demo data
            }
        }
        
        setLoading(false);
    };

    useEffect(() => {
        if (isDemo || sessionId?.startsWith('demo-')) {
            // Demo mode - load pre-made GLB
            loadDemoData();
        } else if (sessionId) {
            loadDataForSession(sessionId);
        } else {
            // Try to load from localStorage
            const savedSession = localStorage.getItem('currentSessionId');
            if (savedSession) {
                if (savedSession.startsWith('demo-')) {
                    loadDemoData();
                } else {
                    loadDataForSession(savedSession);
                }
            } else {
                setLoading(false);
                setError('No session found. Please upload MRI scans first.');
            }
        }
    }, [sessionId, isDemo]);

    const loadData = () => {
        if (isDemo || sessionId?.startsWith('demo-')) {
            loadDemoData();
        } else if (sessionId) {
            loadDataForSession(sessionId);
        }
    };

    const handleDownloadPDF = async () => {
        if (!report) return;

        // Generate HTML report
        const patientName = report.patient_info?.name || report.patient?.name || 'Unknown';
        const patientAge = report.patient_info?.age || report.patient?.age || 'N/A';
        
        const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Brain Tumor Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }
        h1 { color: #314EE6; border-bottom: 2px solid #314EE6; padding-bottom: 10px; }
        h2 { color: #333; margin-top: 30px; }
        .section { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px; }
        .stat { display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px; background: #314EE6; color: white; border-radius: 5px; }
        .finding { padding: 8px 0; border-bottom: 1px solid #ddd; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }
        .header-info { display: flex; justify-content: space-between; margin-bottom: 20px; }
        .warning { background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🧠 Brain Tumor Analysis Report</h1>
    
    <div class="header-info">
        <div>
            <strong>Patient:</strong> ${patientName}<br>
            <strong>Age:</strong> ${patientAge}
        </div>
        <div>
            <strong>Report Date:</strong> ${new Date().toLocaleDateString()}<br>
            <strong>Report ID:</strong> ${report.report_id || 'DEMO-001'}
        </div>
    </div>

    <h2>Tumor Analysis Summary</h2>
    <div class="section">
        <div class="stat">Whole Tumor: ${report.tumor_analysis.whole_tumor_volume_cm3?.toFixed(2) || 'N/A'} cm³</div>
        <div class="stat">Tumor Core: ${report.tumor_analysis.tumor_core_volume_cm3?.toFixed(2) || 'N/A'} cm³</div>
        <div class="stat">Enhancing: ${report.tumor_analysis.enhancing_tumor_volume_cm3?.toFixed(2) || 'N/A'} cm³</div>
    </div>

    <h2>Location</h2>
    <div class="section">
        <strong>Region:</strong> ${report.tumor_analysis.estimated_location?.region || 'N/A'}<br>
        <strong>Hemisphere:</strong> ${report.tumor_analysis.estimated_location?.hemisphere || 'N/A'}
    </div>

    <h2>Grade Assessment</h2>
    <div class="section">
        <strong>Grade:</strong> ${report.tumor_analysis.estimated_grade?.grade || 'N/A'}<br>
        <strong>Confidence:</strong> ${report.tumor_analysis.estimated_grade?.confidence || 'N/A'}<br>
        <p>${report.tumor_analysis.estimated_grade?.description || ''}</p>
    </div>

    <h2>Clinical Findings</h2>
    <div class="section">
        ${report.clinical_findings?.map(f => `<div class="finding">• ${f}</div>`).join('') || '<p>No findings available</p>'}
    </div>

    <h2>Recommendations</h2>
    <div class="section">
        ${report.recommendations?.map(r => `<div class="finding">• ${r}</div>`).join('') || '<p>No recommendations available</p>'}
    </div>

    <div class="warning">
        <strong>⚠️ Disclaimer:</strong> ${report.disclaimer || 'This report is generated by an AI system and should be reviewed by a qualified medical professional before making any clinical decisions.'}
    </div>

    <div class="footer">
        Generated by BraTS Brain Tumor Analysis System<br>
        ${new Date().toLocaleString()}
    </div>
</body>
</html>`;

        // Create a blob and download
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `brain_tumor_report_${report.report_id || 'demo'}.html`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const handleDownloadGLTF = async () => {
        // If we have STL manifest, download the STL files
        if (stlManifest) {
            // Download each STL file
            for (const region of stlManifest.regions) {
                const response = await fetch(`/${region.file}`);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = region.file;
                a.click();
                URL.revokeObjectURL(url);
                // Small delay between downloads
                await new Promise(resolve => setTimeout(resolve, 300));
            }
            return;
        }

        // If we have GLB URL, download it directly
        if (glbUrl) {
            const response = await fetch(glbUrl);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'brain_model.glb';
            a.click();
            URL.revokeObjectURL(url);
            return;
        }

        // Fall back to API download
        const sid = sessionId || localStorage.getItem('currentSessionId');
        if (!sid) return;

        try {
            const blob = await apiService.downloadGLTF(sid);
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `brain_model_${sid}.gltf`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('Failed to download 3D model:', err);
        }
    };

    return (
        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
            <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
                
                {/* Main Area: 3D Viewer */}
                <Box sx={{ width: { xs: '100%', md: '75%' } }}>
                    <Paper 
                        elevation={3}
                        sx={{ 
                            p: 0, 
                            height: '80vh', 
                            borderRadius: 2,
                            border: '2px solid',
                            borderColor: 'primary.main',
                            boxShadow: (theme) => `0 0 12px ${theme.palette.primary.main}`,
                            overflow: 'hidden'
                        }}
                    >
                        {error ? (
                            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', p: 3 }}>
                                <Typography color="error" gutterBottom>{error}</Typography>
                                <Button variant="outlined" startIcon={<RefreshIcon />} onClick={loadData} sx={{ mt: 2 }}>
                                    Retry
                                </Button>
                            </Box>
                        ) : (
                            <BrainViewer 
                                meshData={meshData}
                                glbUrl={glbUrl}
                                stlManifest={stlManifest}
                                stlBaseUrl=""
                                loading={loading}
                                showBrain={showBrain}
                                showNCR={showNCR}
                                showED={showED}
                                showET={showET}
                                brainOpacity={brainOpacity / 100}
                                scale={scale / 100}
                            />
                        )}
                    </Paper>
                </Box>

                {/* Right Panel: Controls & Report */}
                <Box sx={{ width: { xs: '100%', md: '25%' } }}>
                    <Paper 
                        elevation={3}
                        sx={{ 
                            p: 2, 
                            height: '80vh', 
                            overflow: 'auto',
                            borderRadius: 2,
                            border: '2px solid',
                            borderColor: 'primary.main',
                            boxShadow: (theme) => `0 0 12px ${theme.palette.primary.main}`
                        }}
                    >
                        {/* View Controls */}
                        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                            View Controls
                        </Typography>
                        
                        <Box mt={2}>
                            <Typography variant="body2" gutterBottom>Scale</Typography>
                            <Slider 
                                value={scale} 
                                onChange={(_, v) => setScale(v as number)}
                                min={50}
                                max={200}
                                valueLabelDisplay="auto"
                                color="primary" 
                            />
                        </Box>
                        
                        <Box mt={2}>
                            <Typography variant="body2" gutterBottom>Brain Opacity</Typography>
                            <Slider 
                                value={brainOpacity}
                                onChange={(_, v) => setBrainOpacity(v as number)}
                                min={0}
                                max={50}
                                valueLabelDisplay="auto"
                                color="primary" 
                            />
                        </Box>
                        
                        <Box mt={2}>
                            <Typography variant="body2" gutterBottom>Display Layers</Typography>
                            <FormControlLabel 
                                control={<Checkbox checked={showBrain} onChange={(e) => setShowBrain(e.target.checked)} color="primary" size="small" />} 
                                label={<Typography variant="body2">Brain Surface</Typography>}
                            />
                            <FormControlLabel 
                                control={<Checkbox checked={showNCR} onChange={(e) => setShowNCR(e.target.checked)} sx={{ color: '#8B0000', '&.Mui-checked': { color: '#8B0000' } }} size="small" />} 
                                label={<Typography variant="body2">Necrotic Core</Typography>}
                            />
                            <FormControlLabel 
                                control={<Checkbox checked={showED} onChange={(e) => setShowED(e.target.checked)} sx={{ color: '#FFD700', '&.Mui-checked': { color: '#FFD700' } }} size="small" />} 
                                label={<Typography variant="body2">Edema</Typography>}
                            />
                            <FormControlLabel 
                                control={<Checkbox checked={showET} onChange={(e) => setShowET(e.target.checked)} sx={{ color: '#FF0000', '&.Mui-checked': { color: '#FF0000' } }} size="small" />} 
                                label={<Typography variant="body2">Enhancing Tumor</Typography>}
                            />
                        </Box>

                        <Divider sx={{ my: 2 }} />

                        {/* Tumor Statistics */}
                        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                            Tumor Analysis
                        </Typography>
                        
                        {report ? (
                            <>
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                                    <Chip 
                                        label={`WT: ${report.tumor_analysis.whole_tumor_volume_cm3} cm³`}
                                        size="small"
                                        color="primary"
                                    />
                                    <Chip 
                                        label={`TC: ${report.tumor_analysis.tumor_core_volume_cm3} cm³`}
                                        size="small"
                                        color="secondary"
                                    />
                                    <Chip 
                                        label={`ET: ${report.tumor_analysis.enhancing_tumor_volume_cm3} cm³`}
                                        size="small"
                                        sx={{ bgcolor: '#FF0000', color: 'white' }}
                                    />
                                </Box>

                                <Typography variant="body2" color="text.secondary" gutterBottom>
                                    <strong>Location:</strong> {report.tumor_analysis.estimated_location.region} ({report.tumor_analysis.estimated_location.hemisphere})
                                </Typography>

                                <Typography variant="body2" color="text.secondary" gutterBottom>
                                    <strong>Grade:</strong> {report.tumor_analysis.estimated_grade.grade}
                                </Typography>

                                <Typography variant="caption" color="text.secondary">
                                    Confidence: {report.tumor_analysis.estimated_grade.confidence}
                                </Typography>
                            </>
                        ) : loading ? (
                            <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                                <CircularProgress size={24} />
                            </Box>
                        ) : (
                            <Typography variant="body2" color="text.secondary">
                                No analysis data available
                            </Typography>
                        )}

                        <Divider sx={{ my: 2 }} />

                        {/* Download Buttons */}
                        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                            Export
                        </Typography>
                        
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                            <Button 
                                variant="outlined" 
                                startIcon={<DescriptionIcon />}
                                onClick={handleDownloadPDF}
                                disabled={!report}
                                fullWidth
                                size="small"
                            >
                                Download Report
                            </Button>
                            <Button 
                                variant="outlined" 
                                startIcon={<ViewInArIcon />}
                                onClick={handleDownloadGLTF}
                                disabled={!meshData && !stlManifest && !glbUrl}
                                fullWidth
                                size="small"
                            >
                                Download 3D Model
                            </Button>
                        </Box>

                        {report && (
                            <>
                                <Divider sx={{ my: 2 }} />
                                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                                    Report ID: {report.report_id}<br />
                                    Generated: {new Date(report.generated_at).toLocaleString()}
                                </Typography>
                            </>
                        )}
                    </Paper>
                </Box>
            </Box>
        </Container>
    );
}

export default function ViewerPage() {
    return (
        <Suspense fallback={
            <Container maxWidth="xl" sx={{ mt: 4, display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
                <CircularProgress />
            </Container>
        }>
            <ViewerContent />
        </Suspense>
    );
}
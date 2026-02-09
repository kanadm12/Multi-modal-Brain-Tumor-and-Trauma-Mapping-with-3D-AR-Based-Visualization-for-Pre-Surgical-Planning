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

    const loadDemoData = () => {
        // Load demo GLB file and report
        setGlbUrl('/demo_brain_tumor.glb');
        setReport(DEMO_REPORT);
        
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
        const sid = sessionId || localStorage.getItem('currentSessionId');
        if (!sid) return;

        try {
            const blob = await apiService.downloadReportPDF(sid);
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `brain_tumor_report_${sid}.pdf`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('Failed to download PDF:', err);
        }
    };

    const handleDownloadGLTF = async () => {
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
            console.error('Failed to download GLTF:', err);
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
                                Download Report (PDF)
                            </Button>
                            <Button 
                                variant="outlined" 
                                startIcon={<ViewInArIcon />}
                                onClick={handleDownloadGLTF}
                                disabled={!meshData}
                                fullWidth
                                size="small"
                            >
                                Download 3D Model (GLTF)
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
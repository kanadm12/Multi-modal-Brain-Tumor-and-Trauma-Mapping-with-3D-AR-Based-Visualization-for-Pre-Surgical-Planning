"use client";

import React, { useState } from 'react';
import { 
    Box, 
    Button, 
    TextField, 
    Typography, 
    Container, 
    Paper, 
    LinearProgress,
    Alert,
    Snackbar
} from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { useRouter } from 'next/navigation';
import FileUploadZone from '@/components/FileUploadZone';
import ProtectedRoute from '@/components/ProtectedRoute';
import { useAuth } from '@/contexts/AuthContext';
import { apiService } from '@/services/api';
import runpodApi, { RunPodJobResponse } from '@/services/runpod-api';
export const dynamic = 'force-dynamic';

interface PatientDetails {
    name: string;
    age: string;
    weight: string;
    height: string;
    gender: string;
    disorder: string;
    description: string;
}

function UploadPage() {
    const { user } = useAuth();
    const router = useRouter();
    
    const [patientDetails, setPatientDetails] = useState<PatientDetails>({
        name: '',
        age: '',
        weight: '',
        height: '',
        gender: '',
        disorder: '',
        description: ''
    });
    const [files, setFiles] = useState<File[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [statusMessage, setStatusMessage] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [showError, setShowError] = useState(false);

    const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        setPatientDetails(prev => ({ ...prev, [name]: value }));
    };

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            const newFiles = Array.from(event.target.files);
            const niftiFiles = newFiles.filter(f => 
                f.name.endsWith('.nii') || f.name.endsWith('.nii.gz')
            );
            setFiles(niftiFiles);
            
            if (niftiFiles.length !== newFiles.length) {
                setError('Some files were filtered out. Only .nii and .nii.gz files are accepted.');
                setShowError(true);
            }
        }
    };

    const handleProcess = async () => {
        if (files.length === 0) {
            setError('Please upload MRI scan files first.');
            setShowError(true);
            return;
        }

        setIsProcessing(true);
        setProgress(0);
        setError(null);

        try {
            // Check if RunPod is configured
            if (!runpodApi.isConfigured()) {
                throw new Error('RunPod is not configured. Please contact the administrator.');
            }

            setStatusMessage('Creating session...');
            setProgress(5);
            const session = await apiService.createSession({
                patient: {
                    name: patientDetails.name || 'Anonymous Patient',
                    age: patientDetails.age,
                    weight: patientDetails.weight,
                    height: patientDetails.height,
                    disorder: patientDetails.disorder,
                    description: patientDetails.description,
                },
                doctor: {
                    name: user?.full_name || 'Dr. Unknown',
                    email: user?.email || 'doctor@hospital.com',
                    designation: user?.role || 'Doctor',
                    hospital: user?.hospital || 'Hospital',
                },
            });
            localStorage.setItem('currentSessionId', session.session_id);
            localStorage.setItem('patientInfo', JSON.stringify(patientDetails));

            setStatusMessage('Sending MRI scans to AI server...');
            setProgress(15);

            // Submit directly to RunPod (bypasses Vercel size limit)
            const { jobId } = await runpodApi.submitJob(
                files,
                {
                    name: patientDetails.name || 'Anonymous',
                    age: patientDetails.age || 'N/A',
                    id: `PAT-${Date.now()}`,
                },
                {
                    generate_report: true,
                    tta_enabled: false, // Disable TTA for faster inference
                }
            );

            setStatusMessage('AI is analyzing MRI scans...');
            setProgress(25);

            // Poll RunPod for status
            const result = await runpodApi.pollJob(
                jobId,
                (status: RunPodJobResponse) => {
                    let progressValue = 25;
                    switch (status.status) {
                        case 'IN_QUEUE':
                            progressValue = 30;
                            setStatusMessage('Waiting for GPU worker...');
                            break;
                        case 'IN_PROGRESS':
                            progressValue = 50;
                            setStatusMessage('AI is analyzing MRI scans...');
                            break;
                        case 'COMPLETED':
                            progressValue = 95;
                            setStatusMessage('Analysis complete!');
                            break;
                    }
                    setProgress(progressValue);
                },
                3000, // Poll every 3 seconds
                600000 // 10 minute timeout
            );

            // Store result in localStorage for viewer
            localStorage.setItem(`result_${session.session_id}`, JSON.stringify(result));

            setProgress(100);
            setStatusMessage('Analysis complete! Redirecting...');
            setTimeout(() => {
                router.push(`/viewer?session=${session.session_id}`);
            }, 500);

        } catch (err) {
            console.error('Processing error:', err);
            setError(err instanceof Error ? err.message : 'An error occurred during processing');
            setShowError(true);
            setIsProcessing(false);
        }
    };

    const handleCloseError = () => {
        setShowError(false);
    };

    return (
        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
            <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h5" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                    New Analysis
                </Typography>
                <Button
                    variant="outlined"
                    onClick={() => router.push('/dashboard')}
                    disabled={isProcessing}
                    sx={{ borderColor: 'primary.main', color: 'primary.main' }}
                >
                    View History
                </Button>
            </Box>
            
            {isProcessing && (
                <Box sx={{ mb: 3 }}>
                    <Paper sx={{ p: 2, borderRadius: 2, background: 'linear-gradient(135deg, rgba(44, 90, 160, 0.1) 0%, rgba(30, 61, 110, 0.1) 100%)' }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>{statusMessage}</Typography>
                        <LinearProgress variant="determinate" value={progress} sx={{ height: 10, borderRadius: 5, '& .MuiLinearProgress-bar': { background: 'linear-gradient(90deg, #2c5aa0 0%, #4CAF50 100%)' } }} />
                        <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'right' }}>{Math.round(progress)}%</Typography>
                    </Paper>
                </Box>
            )}

            <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
                <Box sx={{ width: { xs: '100%', md: '33.33%' } }}>
                    <Paper elevation={3} sx={{ p: 3, height: '100%', borderRadius: 2, border: '2px solid', borderColor: 'primary.main', boxShadow: (theme) => `0 0 12px ${theme.palette.primary.main}`, opacity: isProcessing ? 0.7 : 1, pointerEvents: isProcessing ? 'none' : 'auto' }}>
                        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>Patient Parameters</Typography>
                        <TextField name="name" label="Patient Name" fullWidth margin="normal" onChange={handleInputChange} value={patientDetails.name} size="small" />
                        <Box sx={{ display: 'flex', gap: 2 }}>
                            <TextField name="age" label="Age" fullWidth margin="normal" onChange={handleInputChange} value={patientDetails.age} size="small" type="number" />
                            <TextField name="gender" label="Gender" fullWidth margin="normal" onChange={handleInputChange} value={patientDetails.gender} size="small" />
                        </Box>
                        <Box sx={{ display: 'flex', gap: 2 }}>
                            <TextField name="weight" label="Weight (kg)" fullWidth margin="normal" onChange={handleInputChange} value={patientDetails.weight} size="small" type="number" />
                            <TextField name="height" label="Height (cm)" fullWidth margin="normal" onChange={handleInputChange} value={patientDetails.height} size="small" type="number" />
                        </Box>
                        <TextField name="disorder" label="Type of Disorder" fullWidth margin="normal" onChange={handleInputChange} value={patientDetails.disorder} size="small" />
                        <TextField name="description" label="Additional Notes" fullWidth margin="normal" multiline rows={4} onChange={handleInputChange} value={patientDetails.description} />
                    </Paper>
                </Box>
                <Box sx={{ width: { xs: '100%', md: '66.67%' } }}>
                    <Paper elevation={3} sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', borderRadius: 2, border: '2px solid', borderColor: 'primary.main', boxShadow: (theme) => `0 0 12px ${theme.palette.primary.main}`, opacity: isProcessing ? 0.7 : 1, pointerEvents: isProcessing ? 'none' : 'auto' }}>
                        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}><CloudUploadIcon sx={{ mr: 1, verticalAlign: 'middle' }} />Upload MRI Scans</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2, textAlign: 'center' }}>Upload the 4 MRI modalities: T1, T1ce, T2, and FLAIR<br />Supported formats: .nii, .nii.gz</Typography>
                        <FileUploadZone onFileSelect={handleFileSelect} fileCount={files.length} />
                        {files.length > 0 && (
                            <Box sx={{ mt: 2, width: '100%', maxHeight: 150, overflow: 'auto' }}>
                                <Typography variant="subtitle2" gutterBottom>Selected files ({files.length}):</Typography>
                                {files.map((file, index) => (
                                    <Typography key={index} variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>• {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)</Typography>
                                ))}
                            </Box>
                        )}
                        <Button variant="contained" size="large" startIcon={isProcessing ? null : <ScienceIcon />} sx={{ mt: 4, width: '50%', py: 1.5, background: 'linear-gradient(45deg, #2c5aa0 30%, #4CAF50 90%)', '&:hover': { background: 'linear-gradient(45deg, #1e3d6e 30%, #388E3C 90%)' } }} onClick={handleProcess} disabled={files.length === 0 || isProcessing}>
                            {isProcessing ? 'Processing...' : 'Analyze with AI'}
                        </Button>
                    </Paper>
                </Box>
            </Box>

            <Snackbar open={showError} autoHideDuration={6000} onClose={handleCloseError} anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
                <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>{error}</Alert>
            </Snackbar>
        </Container>
    );
}

export default function UploadPageWrapper() {
    return (
        <ProtectedRoute>
            <UploadPage />
        </ProtectedRoute>
    );
}
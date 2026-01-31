"use client";

import React from 'react';
import { Paper, Button, Typography, styled } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const DropZone = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  width: '100%',
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: (theme.shape.borderRadius as number) * 2,
  textAlign: 'center',
  transition: 'background-color 0.3s ease, border-color 0.3s ease',
  cursor: 'pointer',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
    borderColor: theme.palette.primary.dark,
  },
}));

interface FileUploadZoneProps {
  onFileSelect: (event: React.ChangeEvent<HTMLInputElement>) => void;
  fileCount: number;
}

const FileUploadZone: React.FC<FileUploadZoneProps> = ({ onFileSelect, fileCount }) => (
  <Button component="label" sx={{ width: '100%', p: 0, textTransform: 'none', mt: 2 }}>
    <DropZone>
      <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
      <Typography variant="h6" gutterBottom>Click to select a folder</Typography>
      <Typography sx={{ color: 'text.secondary' }}>
        {fileCount > 0 ? `${fileCount} files selected` : "Please select the folder containing .nii or .nii.gz files."}
      </Typography>
      <input type="file" hidden multiple webkitdirectory="true" onChange={onFileSelect} />
    </DropZone>
  </Button>
);

export default FileUploadZone;
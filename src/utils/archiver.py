#!/usr/bin/env python3
"""
Archive utility for backing up logs and models before training
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging

class TrainingArchiver:
    """
    Utility class for archiving logs and models before training
    """
    
    def __init__(self, archive_dir: str = "archives"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def create_archive(self, session_name: Optional[str] = None) -> str:
        """
        Create a comprehensive archive of current logs and models
        
        Args:
            session_name: Optional name for the session, otherwise uses timestamp
            
        Returns:
            Path to the created archive file
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        archive_filename = f"training_backup_{session_name}.zip"
        archive_path = self.archive_dir / archive_filename
        
        self.logger.info(f"Creating training archive: {archive_path}")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Archive logs directory
            logs_dir = Path("logs")
            if logs_dir.exists():
                self._add_directory_to_zip(zipf, logs_dir, "logs")
                self.logger.info(f"✓ Archived logs directory")
            
            # Archive models directory
            models_dir = Path("models")
            if models_dir.exists():
                self._add_directory_to_zip(zipf, models_dir, "models")
                self.logger.info(f"✓ Archived models directory")
            
            # Archive config files
            config_dir = Path("config")
            if config_dir.exists():
                self._add_directory_to_zip(zipf, config_dir, "config")
                self.logger.info(f"✓ Archived config directory")
            
            # Archive any checkpoint files in root
            root_checkpoints = list(Path(".").glob("*.pth")) + list(Path(".").glob("*.pkl"))
            for checkpoint in root_checkpoints:
                zipf.write(checkpoint, f"root_checkpoints/{checkpoint.name}")
            
            if root_checkpoints:
                self.logger.info(f"✓ Archived {len(root_checkpoints)} root checkpoint files")
        
        # Get archive size
        archive_size = archive_path.stat().st_size / (1024 * 1024)  # MB
        self.logger.info(f"✓ Archive created successfully: {archive_path} ({archive_size:.1f} MB)")
        
        return str(archive_path)
    
    def _add_directory_to_zip(self, zipf: zipfile.ZipFile, directory: Path, archive_name: str):
        """Add a directory and all its contents to a zip file"""
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                # Create relative path for archive
                relative_path = file_path.relative_to(directory.parent)
                zipf.write(file_path, relative_path)
    
    def cleanup_old_archives(self, keep_count: int = 5):
        """
        Clean up old archive files, keeping only the most recent ones
        
        Args:
            keep_count: Number of recent archives to keep
        """
        archive_files = list(self.archive_dir.glob("training_backup_*.zip"))
        archive_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(archive_files) > keep_count:
            files_to_remove = archive_files[keep_count:]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    self.logger.info(f"✓ Removed old archive: {file_path.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove old archive {file_path.name}: {e}")
    
    def prepare_clean_workspace(self, backup_first: bool = True) -> Optional[str]:
        """
        Prepare a clean workspace for training by archiving existing files
        
        Args:
            backup_first: Whether to create a backup before cleaning
            
        Returns:
            Path to archive if backup was created, None otherwise
        """
        archive_path = None
        
        if backup_first:
            # Check if there's anything worth backing up
            has_logs = Path("logs").exists() and any(Path("logs").iterdir())
            has_models = Path("models").exists() and any(Path("models").iterdir())
            
            if has_logs or has_models:
                archive_path = self.create_archive()
                self.logger.info(f"✓ Created backup before workspace cleanup")
            else:
                self.logger.info("No existing logs or models found, skipping backup")
          # Clean up workspace
        self._clean_workspace()
        
        # Clean up old archives
        self.cleanup_old_archives(keep_count=5)
        
        return archive_path
    
    def _clean_workspace(self):
        """Clean the workspace by completely removing and recreating logs and models directories"""
        try:
            # Remove entire logs directory if it exists
            logs_dir = Path("logs")
            if logs_dir.exists():
                shutil.rmtree(logs_dir)
                self.logger.info("✓ Completely removed existing logs directory")
            
            # Remove entire models directory if it exists
            models_dir = Path("models")
            if models_dir.exists():
                shutil.rmtree(models_dir)
                self.logger.info("✓ Completely removed existing models directory")
            
            # Create fresh empty directories
            logs_dir.mkdir(exist_ok=True)
            (logs_dir / "trades").mkdir(exist_ok=True)
            (logs_dir / "trade_traces").mkdir(exist_ok=True)
            (logs_dir / "tensorboard").mkdir(exist_ok=True)
            models_dir.mkdir(exist_ok=True)
            
            self.logger.info("✓ Created fresh empty logs and models directories")
            self.logger.info("✓ Workspace prepared for new training session with clean directories")
            
        except Exception as e:
            self.logger.error(f"Failed to clean workspace: {e}")
            # Fallback: ensure directories exist
            Path("logs").mkdir(exist_ok=True)
            Path("logs/trades").mkdir(exist_ok=True)
            Path("logs/trade_traces").mkdir(exist_ok=True)
            Path("logs/tensorboard").mkdir(exist_ok=True)
            Path("models").mkdir(exist_ok=True)
            self.logger.info("✓ Ensured basic directory structure exists")

def archive_before_training(session_name: Optional[str] = None) -> str:
    """
    Convenience function to archive before training
    
    Args:
        session_name: Optional session name
        
    Returns:
        Path to created archive
    """
    archiver = TrainingArchiver()
    return archiver.prepare_clean_workspace(backup_first=True)

if __name__ == "__main__":
    # Test the archiver
    print("Testing Training Archiver...")
    
    archiver = TrainingArchiver()
    
    # Create test files if they don't exist
    test_logs_dir = Path("logs")
    test_logs_dir.mkdir(exist_ok=True)
    (test_logs_dir / "test.log").write_text("Test log content")
    
    test_models_dir = Path("models")
    test_models_dir.mkdir(exist_ok=True)
    (test_models_dir / "test_model.pkl").write_text("Test model content")
    
    # Test archiving
    archive_path = archiver.prepare_clean_workspace()
    print(f"Archive created: {archive_path}")
    
    print("Testing completed!")

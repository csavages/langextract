#!/usr/bin/env python3
"""
GUI-based JSONL Sentence Splitter with Auto-Save
Click on text to set split points for entries marked with "split_sentence" label.
Features automatic progress saving and session restoration.
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from typing import List, Dict, Set, Optional, Tuple
import re
from datetime import datetime
import traceback


class JSONLSplitterGUI:
    """GUI application for splitting JSONL entries with split_sentence label"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("JSONL Sentence Splitter - Split Sentence Entries Only")
        self.root.geometry("1200x800")
        
        # Data storage
        self.all_entries = []  # All entries from file
        self.filtered_entries = []  # Entries with split_sentence label
        self.current_index = 0
        self.processed_entries = []
        self.split_positions = set()
        self.next_id = 500001
        self.input_file = None
        self.output_file = None
        self.original_text = ""  # Store original text to avoid corruption
        
        # Progress tracking
        self.progress_file = None
        self.auto_save_interval = 50  # Save every 50 processed entries
        self.last_save_count = 0
        self.session_start_time = datetime.now()
        self.unsaved_changes = False
        
        # Track which original entries have been processed
        self.processed_originals = set()  # Set of _original_line values that have been processed
        
        # Configure styles
        self.setup_styles()
        
        # Create GUI components
        self.create_menu()
        self.create_widgets()
        self.create_status_bar()
        
        # Bind keyboard shortcuts
        self.bind_shortcuts()
        
        # Bind window close event for save on exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize state
        self.update_ui_state()
    
    def setup_styles(self):
        """Configure ttk styles for better appearance"""
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
        style.configure('Split.TLabel', background='yellow', font=('Arial', 10, 'bold'))
        style.configure('FilterInfo.TLabel', font=('Arial', 11, 'bold'), foreground='blue')
        style.configure('Progress.TLabel', font=('Arial', 10, 'italic'), foreground='green')
    
    def create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open JSONL...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Processed...", command=self.save_as, accelerator="Ctrl+S")
        file_menu.add_command(label="Save All (Original + Processed)...", command=self.save_all_with_original, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Save Progress Now", command=self.manual_save_progress, accelerator="Ctrl+P")
        file_menu.add_command(label="Clear Progress", command=self.clear_progress)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Ctrl+Q")
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Split Points", command=self.clear_splits, accelerator="Ctrl+C")
        edit_menu.add_command(label="Auto-Detect Sentences", command=self.auto_detect_sentences, accelerator="Ctrl+D")
        edit_menu.add_separator()
        edit_menu.add_command(label="Settings...", command=self.show_settings)
        
        # Navigation menu
        nav_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Navigation", menu=nav_menu)
        nav_menu.add_command(label="Previous Entry", command=self.prev_entry, accelerator="Left")
        nav_menu.add_command(label="Next Entry", command=self.next_entry, accelerator="Right")
        nav_menu.add_command(label="Go to Entry...", command=self.goto_entry, accelerator="Ctrl+G")
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Statistics", command=self.show_statistics)
        view_menu.add_command(label="Show All Entries", command=self.show_all_entries)
        view_menu.add_command(label="Show Progress Details", command=self.show_progress_details)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_widgets(self):
        """Create main UI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Filter info section
        filter_frame = ttk.Frame(main_frame)
        filter_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.filter_info_label = ttk.Label(filter_frame, 
                                          text="No file loaded", 
                                          style='FilterInfo.TLabel')
        self.filter_info_label.pack(side=tk.LEFT)
        
        # Progress info label
        self.progress_info_label = ttk.Label(filter_frame,
                                            text="",
                                            style='Progress.TLabel')
        self.progress_info_label.pack(side=tk.RIGHT, padx=(20, 0))
        
        # Entry info section
        info_frame = ttk.LabelFrame(main_frame, text="Entry Information", padding="10")
        info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.id_label = ttk.Label(info_frame, text="ID: -", style='Info.TLabel')
        self.id_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.labels_label = ttk.Label(info_frame, text="Labels: -", style='Info.TLabel')
        self.labels_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.manual_label = ttk.Label(info_frame, text="Manual: -", style='Info.TLabel')
        self.manual_label.grid(row=0, column=2, sticky=tk.W)
        
        self.entry_counter_label = ttk.Label(info_frame, text="Entry: 0/0", style='Info.TLabel')
        self.entry_counter_label.grid(row=0, column=3, sticky=tk.E, padx=(20, 0))
        
        info_frame.columnconfigure(3, weight=1)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="Open File", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Previous", command=self.prev_entry).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Next", command=self.next_entry).pack(side=tk.LEFT, padx=2)
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Button(control_frame, text="Clear Splits", command=self.clear_splits).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Auto-Detect", command=self.auto_detect_sentences).pack(side=tk.LEFT, padx=2)
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Button(control_frame, text="Apply & Next", command=self.apply_and_next, 
                  state=tk.DISABLED).pack(side=tk.LEFT, padx=2)
        self.apply_next_btn = control_frame.winfo_children()[-1]  # Store reference
        ttk.Button(control_frame, text="Skip", command=self.skip_entry).pack(side=tk.LEFT, padx=2)
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Button(control_frame, text="Process All Remaining", command=self.process_all_remaining).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Save", command=self.save_as).pack(side=tk.LEFT, padx=2)
        
        # Text display area (left side)
        text_frame = ttk.LabelFrame(main_frame, text="Click to Set Split Points", padding="10")
        text_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Create text widget with scrollbar
        text_scroll = ttk.Scrollbar(text_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_display = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 14), 
                                   height=20, width=50, yscrollcommand=text_scroll.set,
                                   cursor="hand2")
        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.text_display.yview)
        
        # Configure text tags for styling
        self.text_display.tag_configure("split_marker", background="yellow", 
                                       font=('Consolas', 14, 'bold'))
        self.text_display.tag_configure("hover", background="lightblue")
        
        # Bind mouse events
        self.text_display.bind("<Button-1>", self.on_text_click)
        self.text_display.bind("<Motion>", self.on_mouse_motion)
        self.text_display.bind("<Leave>", self.on_mouse_leave)
        
        # Preview area (right side)
        preview_frame = ttk.LabelFrame(main_frame, text="Split Preview", padding="10")
        preview_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Preview with scrollbar
        preview_scroll = ttk.Scrollbar(preview_frame)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, 
                                                      font=('Arial', 11), height=20, 
                                                      width=50, state=tk.DISABLED)
        self.preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # ID assignment info
        id_info_frame = ttk.Frame(preview_frame)
        id_info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        ttk.Label(id_info_frame, text="Next ID:").pack(side=tk.LEFT)
        self.next_id_var = tk.StringVar(value=str(self.next_id))
        id_entry = ttk.Entry(id_info_frame, textvariable=self.next_id_var, width=10)
        id_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(id_info_frame, text="Update", command=self.update_next_id).pack(side=tk.LEFT)
    
    def create_status_bar(self):
        """Create status bar at bottom of window"""
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.status_bar = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Auto-save indicator
        self.auto_save_indicator = ttk.Label(status_frame, text="", relief=tk.SUNKEN, anchor=tk.E, width=30)
        self.auto_save_indicator.pack(side=tk.RIGHT)
    
    def bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-s>', lambda e: self.save_as())
        self.root.bind('<Control-S>', lambda e: self.save_all_with_original())
        self.root.bind('<Control-p>', lambda e: self.manual_save_progress())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
        self.root.bind('<Control-c>', lambda e: self.clear_splits())
        self.root.bind('<Control-d>', lambda e: self.auto_detect_sentences())
        self.root.bind('<Control-g>', lambda e: self.goto_entry())
        self.root.bind('<Left>', lambda e: self.prev_entry())
        self.root.bind('<Right>', lambda e: self.next_entry())
        self.root.bind('<Return>', lambda e: self.apply_and_next())
    
    def update_ui_state(self):
        """Update UI elements based on current state"""
        has_entries = len(self.filtered_entries) > 0
        
        # Update apply & next button
        if hasattr(self, 'apply_next_btn'):
            self.apply_next_btn.config(state=tk.NORMAL if has_entries else tk.DISABLED)
        
        # Update entry counter
        if has_entries:
            self.entry_counter_label.config(
                text=f"Entry: {self.current_index + 1}/{len(self.filtered_entries)}"
            )
        else:
            self.entry_counter_label.config(text="Entry: 0/0")
        
        # Update filter info
        if self.all_entries:
            filtered_count = len(self.filtered_entries)
            total_count = len(self.all_entries)
            self.filter_info_label.config(
                text=f"Showing {filtered_count} entries with 'split_sentence' label out of {total_count} total entries"
            )
        else:
            self.filter_info_label.config(text="No file loaded")
        
        # Update progress info
        self.update_progress_display()
    
    def update_progress_display(self):
        """Update the progress information display"""
        if self.processed_entries:
            # Count unique processed originals
            unique_processed = len(self.processed_originals)
            total_to_process = len(self.filtered_entries)
            
            progress_text = f"Progress: {unique_processed}/{total_to_process} processed"
            if self.unsaved_changes:
                progress_text += " (unsaved changes)"
            
            self.progress_info_label.config(text=progress_text)
            
            # Update auto-save indicator
            entries_since_save = len(self.processed_entries) - self.last_save_count
            if entries_since_save > 0:
                self.auto_save_indicator.config(
                    text=f"Auto-save in {self.auto_save_interval - entries_since_save} entries"
                )
            else:
                self.auto_save_indicator.config(text="Progress saved")
        else:
            self.progress_info_label.config(text="")
            self.auto_save_indicator.config(text="")
    
    def on_closing(self):
        """Handle window closing event - save progress before exit"""
        if self.unsaved_changes and self.processed_entries:
            response = messagebox.askyesnocancel(
                "Save Progress", 
                f"You have {len(self.processed_entries) - self.last_save_count} unsaved processed entries.\n"
                "Save progress before closing?"
            )
            if response is True:  # Yes
                self.save_progress()
                self.root.destroy()
            elif response is False:  # No
                confirm = messagebox.askyesno(
                    "Confirm Exit",
                    "Are you sure you want to exit without saving?\n"
                    "Unsaved progress will be lost."
                )
                if confirm:
                    self.root.destroy()
            # else: Cancel - do nothing
        else:
            self.root.destroy()
    
    def save_progress(self, show_message=True):
        """Save current progress to a progress file"""
        if not self.input_file:
            return False
            
        # Create progress filename
        base_name = self.input_file.replace('.jsonl', '')
        self.progress_file = f"{base_name}_splitter_progress.json"
        
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'session_start': self.session_start_time.isoformat(),
            'input_file': self.input_file,
            'current_index': self.current_index,
            'next_id': self.next_id,
            'total_entries': len(self.all_entries),
            'total_filtered': len(self.filtered_entries),
            'total_processed_entries': len(self.processed_entries),
            'unique_originals_processed': len(self.processed_originals),
            'processed_originals': list(self.processed_originals),
            'processed_entries': []
        }
        
        # Store processed entries
        for entry in self.processed_entries:
            # Store clean entry without temporary fields
            clean_entry = {k: v for k, v in entry.items() if not k.startswith('_')}
            progress_data['processed_entries'].append(clean_entry)
        
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
            self.last_save_count = len(self.processed_entries)
            self.unsaved_changes = False
            
            if show_message:
                self.status_bar.config(text=f"Progress saved ({len(self.processed_entries)} entries)")
            
            self.update_progress_display()
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save progress:\n{str(e)}")
            return False
    
    def load_progress(self):
        """Load previous progress if it exists"""
        if not self.input_file:
            return False
            
        base_name = self.input_file.replace('.jsonl', '')
        progress_file = f"{base_name}_splitter_progress.json"
        
        if os.path.exists(progress_file):
            response = messagebox.askyesno(
                "Load Progress",
                f"Found previous progress file with saved work.\n"
                f"Would you like to continue from where you left off?"
            )
            if response:
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress_data = json.load(f)
                    
                    # Verify it's for the same input file
                    if progress_data.get('input_file') != self.input_file:
                        messagebox.showwarning(
                            "Warning",
                            "Progress file is for a different input file.\n"
                            "Starting fresh session."
                        )
                        return False
                    
                    # Restore state
                    self.processed_entries = progress_data.get('processed_entries', [])
                    self.current_index = progress_data.get('current_index', 0)
                    self.next_id = progress_data.get('next_id', 500001)
                    self.next_id_var.set(str(self.next_id))
                    self.processed_originals = set(progress_data.get('processed_originals', []))
                    self.last_save_count = len(self.processed_entries)
                    
                    # Update filtered entries to skip already processed ones
                    remaining_filtered = []
                    for entry in self.filtered_entries:
                        if entry.get('_original_line') not in self.processed_originals:
                            remaining_filtered.append(entry)
                    
                    if remaining_filtered:
                        self.filtered_entries = remaining_filtered
                        self.current_index = 0
                    
                    # Show summary
                    unique_processed = len(self.processed_originals)
                    total_processed_entries = len(self.processed_entries)
                    remaining = len(self.filtered_entries)
                    
                    messagebox.showinfo(
                        "Progress Loaded",
                        f"Loaded previous session:\n"
                        f"• {unique_processed} original entries processed\n"
                        f"• {total_processed_entries} total entries created\n"
                        f"• {remaining} entries remaining to process"
                    )
                    
                    self.status_bar.config(
                        text=f"Loaded {total_processed_entries} processed entries from previous session"
                    )
                    
                    # Update UI
                    if self.filtered_entries:
                        self.display_current_entry()
                    self.update_ui_state()
                    
                    return True
                    
                except Exception as e:
                    messagebox.showerror(
                        "Error",
                        f"Failed to load progress:\n{str(e)}\n\n"
                        "Starting fresh session."
                    )
                    if messagebox.askyesno("Debug Info", "Show detailed error?"):
                        messagebox.showinfo("Error Details", traceback.format_exc())
                    return False
        
        return False
    
    def manual_save_progress(self):
        """Manually save progress when user requests it"""
        if self.processed_entries:
            if self.save_progress(show_message=True):
                messagebox.showinfo(
                    "Progress Saved",
                    f"Progress saved successfully!\n"
                    f"• {len(self.processed_entries)} total entries saved\n"
                    f"• {len(self.processed_originals)} original entries processed"
                )
        else:
            messagebox.showinfo("No Progress", "No processed entries to save yet.")
    
    def check_auto_save(self):
        """Check if auto-save should be triggered"""
        entries_since_save = len(self.processed_entries) - self.last_save_count
        if entries_since_save >= self.auto_save_interval:
            self.save_progress(show_message=False)
            self.status_bar.config(text=f"Auto-saved progress ({len(self.processed_entries)} entries)")
    
    def clear_progress(self):
        """Clear the progress file after confirmation"""
        if not self.progress_file or not os.path.exists(self.progress_file):
            messagebox.showinfo("No Progress File", "No progress file to clear.")
            return
        
        response = messagebox.askyesno(
            "Clear Progress",
            "Are you sure you want to clear the saved progress?\n"
            "This action cannot be undone."
        )
        
        if response:
            try:
                os.remove(self.progress_file)
                self.status_bar.config(text="Progress file cleared")
                messagebox.showinfo("Success", "Progress file has been cleared.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear progress file:\n{str(e)}")
    
    def show_settings(self):
        """Show settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Auto-save interval setting
        ttk.Label(dialog, text="Auto-Save Settings", font=('Arial', 12, 'bold')).pack(pady=10)
        
        frame = ttk.Frame(dialog)
        frame.pack(pady=10)
        
        ttk.Label(frame, text="Auto-save after every").grid(row=0, column=0, padx=5)
        
        interval_var = tk.StringVar(value=str(self.auto_save_interval))
        interval_entry = ttk.Entry(frame, textvariable=interval_var, width=10)
        interval_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(frame, text="processed entries").grid(row=0, column=2, padx=5)
        
        # Next ID setting
        ttk.Label(dialog, text="ID Assignment", font=('Arial', 12, 'bold')).pack(pady=(20, 10))
        
        id_frame = ttk.Frame(dialog)
        id_frame.pack(pady=10)
        
        ttk.Label(id_frame, text="Starting ID for new entries:").grid(row=0, column=0, padx=5)
        
        id_var = tk.StringVar(value=str(self.next_id))
        id_entry = ttk.Entry(id_frame, textvariable=id_var, width=15)
        id_entry.grid(row=0, column=1, padx=5)
        
        def apply_settings():
            try:
                new_interval = int(interval_var.get())
                if new_interval < 1:
                    raise ValueError("Interval must be at least 1")
                self.auto_save_interval = new_interval
                
                new_id = int(id_var.get())
                if new_id < 1:
                    raise ValueError("ID must be positive")
                self.next_id = new_id
                self.next_id_var.set(str(self.next_id))
                
                self.status_bar.config(text="Settings updated")
                dialog.destroy()
                
            except ValueError as e:
                messagebox.showerror("Invalid Settings", str(e))
        
        ttk.Button(dialog, text="Apply", command=apply_settings).pack(pady=20)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack()
    
    def show_progress_details(self):
        """Show detailed progress information"""
        if not self.processed_entries and not self.progress_file:
            messagebox.showinfo("No Progress", "No progress to show.")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Progress Details")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        
        # Create text widget for details
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, width=70, height=25)
        text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Generate progress report
        report = f"Progress Report\n"
        report += f"{'='*60}\n\n"
        
        report += f"Session Information:\n"
        report += f"  Started: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"  Input File: {os.path.basename(self.input_file) if self.input_file else 'None'}\n"
        report += f"  Progress File: {os.path.basename(self.progress_file) if self.progress_file else 'None'}\n\n"
        
        report += f"Statistics:\n"
        report += f"  Total entries in file: {len(self.all_entries)}\n"
        report += f"  Entries needing splits: {len(self.filtered_entries) + len(self.processed_originals)}\n"
        report += f"  Entries processed: {len(self.processed_originals)}\n"
        report += f"  Entries remaining: {len(self.filtered_entries)}\n"
        report += f"  Total new entries created: {len(self.processed_entries)}\n\n"
        
        report += f"Auto-Save Settings:\n"
        report += f"  Interval: Every {self.auto_save_interval} entries\n"
        report += f"  Last saved: {self.last_save_count} entries\n"
        report += f"  Unsaved changes: {len(self.processed_entries) - self.last_save_count} entries\n\n"
        
        if self.processed_entries:
            report += f"Recent Processed Entries (last 10):\n"
            report += f"{'-'*60}\n"
            for entry in self.processed_entries[-10:]:
                report += f"  ID: {entry.get('id', 'N/A')}\n"
                text_preview = entry.get('suggested_imaging_observation_sentence', '')[:100]
                if len(entry.get('suggested_imaging_observation_sentence', '')) > 100:
                    text_preview += "..."
                report += f"  Text: {text_preview}\n"
                report += f"  Labels: {', '.join(entry.get('label', []))}\n"
                report += f"{'-'*40}\n"
        
        text.insert(tk.END, report)
        text.config(state=tk.DISABLED)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def open_file(self):
        """Open and load a JSONL file"""
        # Check for unsaved changes
        if self.unsaved_changes and self.processed_entries:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Save before opening a new file?"
            )
            if response is True:  # Yes
                self.save_progress()
            elif response is None:  # Cancel
                return
        
        filename = filedialog.askopenfilename(
            title="Open JSONL File",
            filetypes=[("JSONL files", "*.jsonl"), ("JSON files", "*.json"), 
                      ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.load_jsonl(filename)
            self.input_file = filename
            
            # Reset processing state
            self.processed_entries = []
            self.processed_originals = set()
            self.last_save_count = 0
            self.unsaved_changes = False
            self.session_start_time = datetime.now()
            
            # Filter entries with split_sentence label
            self.filter_entries()
            
            # Try to load previous progress
            if self.load_progress():
                # Progress was loaded, filtered_entries already updated
                pass
            elif not self.filtered_entries:
                messagebox.showwarning("No Matching Entries", 
                                      f"No entries with 'split_sentence' label found in {os.path.basename(filename)}")
                self.status_bar.config(text=f"Loaded {len(self.all_entries)} entries, 0 need splitting")
            else:
                self.status_bar.config(text=f"Loaded {len(self.filtered_entries)} entries to split from {os.path.basename(filename)}")
                self.current_index = 0
                self.display_current_entry()
            
            self.update_ui_state()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def load_jsonl(self, filepath: str):
        """Load entries from JSONL file"""
        self.all_entries = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    # Store original line number for reference
                    entry['_original_line'] = line_num
                    self.all_entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
    
    def filter_entries(self):
        """Filter entries to only include those with split_sentence label"""
        self.filtered_entries = []
        
        for entry in self.all_entries:
            labels = entry.get('label', [])
            if 'split_sentence' in labels:
                self.filtered_entries.append(entry)
    
    def save_as(self):
        """Save only processed split_sentence entries"""
        if not self.processed_entries:
            messagebox.showwarning("Warning", "No processed entries to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Processed Entries",
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )
        
        if filename:
            self.output_file = filename
            if self.save_to_file(filename, self.processed_entries):
                # Ask about clearing progress
                if self.progress_file and os.path.exists(self.progress_file):
                    response = messagebox.askyesno(
                        "Clear Progress",
                        "Save successful! Would you like to clear the progress file?"
                    )
                    if response:
                        self.clear_progress()
    
    def save_all_with_original(self):
        """Save all entries including non-split_sentence ones and processed splits"""
        if not self.all_entries:
            messagebox.showwarning("Warning", "No entries loaded")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save All Entries (Original + Processed)",
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )
        
        if filename:
            # Combine non-split_sentence entries with processed ones
            combined_entries = []
            
            # Add processed entries
            for entry in self.processed_entries:
                combined_entries.append(entry)
            
            # Add unprocessed entries
            for entry in self.all_entries:
                labels = entry.get('label', [])
                if 'split_sentence' not in labels:
                    # Keep all non-split_sentence entries
                    combined_entries.append(entry)
                elif entry.get('_original_line') not in self.processed_originals:
                    # Keep unprocessed split_sentence entries
                    combined_entries.append(entry)
            
            # Sort by original line number if available
            combined_entries.sort(key=lambda x: x.get('_original_line', float('inf')))
            
            if self.save_to_file(filename, combined_entries):
                self.status_bar.config(text=f"Saved {len(combined_entries)} total entries to {os.path.basename(filename)}")
                
                # Ask about clearing progress
                if self.progress_file and os.path.exists(self.progress_file):
                    response = messagebox.askyesno(
                        "Clear Progress",
                        "Save successful! Would you like to clear the progress file?"
                    )
                    if response:
                        self.clear_progress()
    
    def process_all_remaining(self):
        """Process all remaining filtered entries without user interaction"""
        if not self.filtered_entries:
            messagebox.showwarning("Warning", "No entries to process")
            return
        
        # Confirm action
        remaining = len(self.filtered_entries) - self.current_index
        if self.split_positions:
            remaining += 1  # Include current entry
        
        if remaining > 20:  # Warn for large batches
            response = messagebox.askyesno(
                "Confirm Batch Processing",
                f"This will auto-process {remaining} entries.\n"
                "Are you sure you want to continue?"
            )
            if not response:
                return
        
        processed_count = 0
        
        # Process current entry if it has splits
        if self.split_positions:
            self.apply_splits()
            processed_count += 1
        
        # Process remaining entries
        start_index = self.current_index + 1 if not self.split_positions else self.current_index
        for i in range(start_index, len(self.filtered_entries)):
            self.current_index = i
            entry = self.filtered_entries[i]
            
            # Skip if already processed
            if entry.get('_original_line') in self.processed_originals:
                continue
            
            # Auto-detect and apply splits
            text = entry.get('suggested_imaging_observation_sentence', '')
            if text:
                positions = self._auto_detect_positions(text)
                if positions:
                    self._apply_splits_to_entry(entry, positions)
                    processed_count += 1
                else:
                    # No automatic splits found, keep original
                    self.processed_entries.append(entry.copy())
                    self.processed_originals.add(entry.get('_original_line'))
                    processed_count += 1
                    self.unsaved_changes = True
            
            # Check for auto-save
            self.check_auto_save()
            
            # Update UI periodically
            if processed_count % 10 == 0:
                self.update_progress_display()
                self.root.update_idletasks()
        
        # Final save
        self.save_progress(show_message=False)
        
        self.status_bar.config(text=f"Batch processed {processed_count} entries")
        messagebox.showinfo("Processing Complete", 
                           f"Processed {processed_count} entries.\n"
                           f"Total processed entries: {len(self.processed_entries)}")
        
        # Move to the last entry or reset if all done
        if self.current_index >= len(self.filtered_entries) - 1:
            self.current_index = len(self.filtered_entries) - 1
        self.display_current_entry()
    
    def save_to_file(self, filepath: str, entries: List[Dict]) -> bool:
        """Save entries to file"""
        try:
            # Remove temporary fields
            clean_entries = []
            for entry in entries:
                clean_entry = {k: v for k, v in entry.items() if not k.startswith('_')}
                clean_entries.append(clean_entry)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for entry in clean_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            messagebox.showinfo("Success", f"Saved {len(clean_entries)} entries successfully")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")
            return False
    
    def display_current_entry(self):
        """Display the current entry in the UI"""
        if not self.filtered_entries or self.current_index >= len(self.filtered_entries):
            # No more entries to display
            self.text_display.config(state=tk.NORMAL)
            self.text_display.delete('1.0', tk.END)
            self.text_display.insert('1.0', "No more entries to process.")
            self.text_display.config(state=tk.DISABLED)
            self.original_text = ""
            return
        
        entry = self.filtered_entries[self.current_index]
        
        # Update info labels
        self.id_label.config(text=f"ID: {entry.get('id', 'N/A')}")
        labels = entry.get('label', [])
        labels_text = ', '.join(labels) if labels else 'None'
        self.labels_label.config(text=f"Labels: {labels_text}")
        self.manual_label.config(text=f"Manual: {entry.get('manual', 'N/A')}")
        
        # Display text and store original
        text = entry.get('suggested_imaging_observation_sentence', '')
        self.original_text = text  # Store original text
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete('1.0', tk.END)
        self.text_display.insert('1.0', text)
        
        # Clear splits for new entry
        self.split_positions.clear()
        self.update_split_display()
        
        # Update counter
        self.update_ui_state()
    
    def on_text_click(self, event):
        """Handle mouse click on text to add/remove split point"""
        try:
            # Get click position
            index = self.text_display.index(f"@{event.x},{event.y}")
            position = self.text_display.count('1.0', index, 'chars')[0]
            
            # Get actual text length (without including the markers)
            text_length = len(self.original_text)
            
            # Adjust position for existing markers
            adjusted_position = position
            for split_pos in sorted(self.split_positions):
                if split_pos < position:
                    adjusted_position -= 1  # Account for the pipe character
            
            # Check bounds
            if adjusted_position > text_length:
                return
            
            # Toggle split point
            if adjusted_position in self.split_positions:
                self.split_positions.remove(adjusted_position)
            else:
                self.split_positions.add(adjusted_position)
            
            self.update_split_display()
        except Exception as e:
            print(f"Click error: {e}")
    
    def on_mouse_motion(self, event):
        """Handle mouse hover to show potential split point"""
        # Remove previous hover tag
        self.text_display.tag_remove("hover", '1.0', tk.END)
        
        # Get mouse position in text
        index = self.text_display.index(f"@{event.x},{event.y}")
        
        # Add hover effect at character position
        self.text_display.tag_add("hover", index)
    
    def on_mouse_leave(self, event):
        """Remove hover effect when mouse leaves"""
        self.text_display.tag_remove("hover", '1.0', tk.END)
    
    def update_split_display(self):
        """Update the display to show split markers and preview"""
        # Clear existing markers
        self.text_display.tag_remove("split_marker", '1.0', tk.END)
        
        # Use stored original text
        text = self.original_text
        
        # Clear and redisplay original text
        self.text_display.delete('1.0', tk.END)
        self.text_display.insert('1.0', text)
        
        # Add split markers
        for position in sorted(self.split_positions, reverse=True):
            if position <= len(text):
                index = f"1.0 + {position} chars"
                self.text_display.insert(index, '|')
                self.text_display.tag_add("split_marker", index)
        
        # Update preview
        self.update_preview(text)
    
    def update_preview(self, text: str):
        """Update the preview pane with split segments"""
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete('1.0', tk.END)
        
        if not self.split_positions:
            self.preview_text.insert(tk.END, "No split points set.\n\n")
            self.preview_text.insert(tk.END, "Original text:\n")
            self.preview_text.insert(tk.END, f'ID: {self.next_id}\n')
            self.preview_text.insert(tk.END, f'Text: "{text}"')
        else:
            # Create segments
            positions = sorted(self.split_positions)
            segments = []
            start = 0
            
            for pos in positions:
                if start < pos:
                    segments.append(text[start:pos])
                start = pos
            
            # Add final segment
            if start < len(text):
                segments.append(text[start:])
            
            # Display segments
            self.preview_text.insert(tk.END, f"Will create {len(segments)} entries:\n\n")
            
            current_id = self.next_id
            for i, segment in enumerate(segments, 1):
                if segment.strip():  # Only show non-empty segments
                    self.preview_text.insert(tk.END, f"Entry {i}:\n")
                    self.preview_text.insert(tk.END, f"  ID: {current_id}\n")
                    self.preview_text.insert(tk.END, f'  Text: "{segment}"\n\n')
                    current_id += 1
        
        self.preview_text.config(state=tk.DISABLED)
    
    def clear_splits(self):
        """Clear all split points"""
        self.split_positions.clear()
        self.display_current_entry()
    
    def auto_detect_sentences(self):
        """Automatically detect sentence boundaries"""
        text = self.original_text  # Use stored original text
        
        positions = self._auto_detect_positions(text)
        
        if positions:
            self.split_positions.update(positions)
            self.update_split_display()
            self.status_bar.config(text=f"Auto-detected {len(positions)} sentence boundaries")
        else:
            self.status_bar.config(text="No sentence boundaries detected")
    
    def _auto_detect_positions(self, text: str) -> List[int]:
        """Internal method to detect sentence boundaries"""
        # Pattern for sentence endings
        pattern = r'[.!?]+(?:\s+|$)'
        
        positions = []
        for match in re.finditer(pattern, text):
            if match.end() < len(text):
                positions.append(match.end())
        
        return positions
    
    def get_next_safe_id(self):
        """Get next ID that doesn't collide with existing entries"""
        existing_ids = set()
        for entry in self.all_entries + self.processed_entries:
            if 'id' in entry:
                existing_ids.add(entry['id'])
        
        while self.next_id in existing_ids:
            self.next_id += 1
        
        return self.next_id
    
    def apply_splits(self):
        """Apply current splits and add to processed entries"""
        if not self.filtered_entries or self.current_index >= len(self.filtered_entries):
            return
        
        entry = self.filtered_entries[self.current_index]
        self._apply_splits_to_entry(entry, self.split_positions)
        self.status_bar.config(text=f"Processed entry {self.current_index + 1}")
        
        # Mark as having unsaved changes
        self.unsaved_changes = True
        
        # Check for auto-save
        self.check_auto_save()
    
    def _apply_splits_to_entry(self, entry: Dict, positions: Set[int]):
        """Internal method to apply splits to an entry"""
        text = entry.get('suggested_imaging_observation_sentence', '')
        
        # Track that we've processed this original entry
        self.processed_originals.add(entry.get('_original_line'))
        
        if not positions:
            # No splits, keep original
            processed = entry.copy()
            self.processed_entries.append(processed)
        else:
            # Create split entries
            positions = sorted(positions)
            segments = []
            start = 0
            
            for pos in positions:
                if start < pos:
                    segments.append(text[start:pos])
                start = pos
            
            # Add final segment
            if start < len(text):
                segments.append(text[start:])
            
            # Create new entries
            for segment in segments:
                if segment.strip():
                    new_entry = entry.copy()
                    new_entry['id'] = self.get_next_safe_id()
                    new_entry['suggested_imaging_observation_sentence'] = segment
                    self.processed_entries.append(new_entry)
                    self.next_id += 1
    
    def apply_and_next(self):
        """Apply splits and move to next entry"""
        if not self.filtered_entries:
            return
            
        self.apply_splits()
        self.next_entry()
    
    def skip_entry(self):
        """Skip current entry without processing"""
        if not self.filtered_entries or self.current_index >= len(self.filtered_entries):
            return
        
        # Add original entry without modifications
        entry = self.filtered_entries[self.current_index].copy()
        self.processed_entries.append(entry)
        self.processed_originals.add(entry.get('_original_line'))
        
        # Mark as having unsaved changes
        self.unsaved_changes = True
        
        # Check for auto-save
        self.check_auto_save()
        
        self.next_entry()
    
    def prev_entry(self):
        """Navigate to previous entry"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_entry()
    
    def next_entry(self):
        """Navigate to next entry"""
        if self.current_index < len(self.filtered_entries) - 1:
            self.current_index += 1
            self.display_current_entry()
        else:
            messagebox.showinfo("End of Filtered Entries", 
                               f"Reached the last entry with 'split_sentence' label.\n"
                               f"Processed {len(self.processed_entries)} entries so far.")
    
    def goto_entry(self):
        """Jump to specific entry number"""
        if not self.filtered_entries:
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Go to Entry")
        dialog.geometry("300x100")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"Enter entry number (1-{len(self.filtered_entries)}):").pack(pady=10)
        
        entry_var = tk.StringVar()
        entry_widget = ttk.Entry(dialog, textvariable=entry_var)
        entry_widget.pack(pady=5)
        entry_widget.focus()
        
        def go():
            try:
                num = int(entry_var.get()) - 1
                if 0 <= num < len(self.filtered_entries):
                    self.current_index = num
                    self.display_current_entry()
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Invalid entry number")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")
        
        ttk.Button(dialog, text="Go", command=go).pack(pady=10)
        entry_widget.bind('<Return>', lambda e: go())
    
    def update_next_id(self):
        """Update the next ID counter with validation"""
        try:
            new_id = int(self.next_id_var.get())
            if new_id < 1:
                raise ValueError("ID must be positive")
            
            # Check for conflicts with existing IDs
            existing_ids = {e.get('id') for e in self.all_entries + self.processed_entries if 'id' in e}
            if new_id in existing_ids:
                if not messagebox.askyesno("ID Conflict", f"ID {new_id} already exists. Continue anyway?"):
                    self.next_id_var.set(str(self.next_id))
                    return
            
            self.next_id = new_id
            self.status_bar.config(text=f"Updated next ID to {self.next_id}")
            self.update_split_display()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.next_id_var.set(str(self.next_id))
    
    def show_statistics(self):
        """Show statistics about the loaded file"""
        if not self.all_entries:
            messagebox.showinfo("Statistics", "No file loaded")
            return
        
        # Calculate statistics
        total = len(self.all_entries)
        split_needed = len(self.filtered_entries) + len(self.processed_originals)
        processed = len(self.processed_originals)
        remaining = len(self.filtered_entries)
        
        # Count label distribution
        label_counts = {}
        for entry in self.all_entries:
            for label in entry.get('label', []):
                label_counts[label] = label_counts.get(label, 0) + 1
        
        stats_text = f"""File Statistics:
        
Total entries: {total}
Entries with 'split_sentence' label: {split_needed}
Entries without 'split_sentence' label: {total - split_needed}

Processing Status:
Processed originals: {processed}
New entries created: {len(self.processed_entries)}
Remaining to process: {remaining}

Session Information:
Session started: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}
Auto-save interval: Every {self.auto_save_interval} entries
Unsaved changes: {len(self.processed_entries) - self.last_save_count} entries

Label Distribution:
"""
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            stats_text += f"  {label}: {count}\n"
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Statistics")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, width=50, height=25)
        text.pack(padx=10, pady=10)
        text.insert(tk.END, stats_text)
        text.config(state=tk.DISABLED)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
    
    def show_all_entries(self):
        """Show all entries in the file for reference"""
        if not self.all_entries:
            messagebox.showinfo("All Entries", "No file loaded")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("All Entries")
        dialog.geometry("900x600")
        dialog.transient(self.root)
        
        # Create treeview for displaying entries
        tree_frame = ttk.Frame(dialog)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, 
                           columns=('ID', 'Labels', 'Status', 'Text'), show='tree headings')
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.config(command=tree.yview)
        
        # Configure columns
        tree.heading('#0', text='#')
        tree.heading('ID', text='ID')
        tree.heading('Labels', text='Labels')
        tree.heading('Status', text='Status')
        tree.heading('Text', text='Text')
        
        tree.column('#0', width=50)
        tree.column('ID', width=100)
        tree.column('Labels', width=200)
        tree.column('Status', width=100)
        tree.column('Text', width=400)
        
        # Add entries
        for i, entry in enumerate(self.all_entries, 1):
            labels = ', '.join(entry.get('label', []))
            text = entry.get('suggested_imaging_observation_sentence', '')[:100]  # Truncate long text
            has_split = 'split_sentence' in entry.get('label', [])
            
            # Determine status
            if has_split:
                if entry.get('_original_line') in self.processed_originals:
                    status = "Processed"
                    tag = 'processed'
                else:
                    status = "Pending"
                    tag = 'pending'
            else:
                status = "N/A"
                tag = 'normal'
            
            # Add with appropriate tag
            tree.insert('', 'end', text=str(i), 
                       values=(entry.get('id', 'N/A'), labels, status, text),
                       tags=(tag,))
        
        # Configure tags for visual distinction
        tree.tag_configure('processed', background='lightgreen')
        tree.tag_configure('pending', background='lightyellow')
        tree.tag_configure('normal', background='white')
        
        # Add summary label
        summary_label = ttk.Label(dialog, 
                                text=f"Total: {len(self.all_entries)} | "
                                     f"Processed: {len(self.processed_originals)} | "
                                     f"Pending: {len(self.filtered_entries)}")
        summary_label.pack(pady=5)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
    
    def show_instructions(self):
        """Show help instructions"""
        instructions = """
JSONL Sentence Splitter - Instructions

This tool processes ONLY entries with the "split_sentence" label.

1. LOADING FILES:
   - Click "Open File" or use Ctrl+O to load a JSONL file
   - The tool automatically filters entries with "split_sentence" label
   - If a progress file exists, you'll be asked to continue from where you left off

2. SETTING SPLIT POINTS:
   - Click anywhere in the text to add a split point
   - Click on an existing split point (yellow highlight) to remove it
   - Split points are shown with yellow highlighting and "|" markers

3. AUTO-DETECTION:
   - Click "Auto-Detect" or press Ctrl+D to automatically find sentence boundaries
   - The tool looks for periods, exclamation marks, and question marks

4. NAVIGATION:
   - Use "Previous" / "Next" buttons or arrow keys to navigate filtered entries
   - "Go to Entry" (Ctrl+G) to jump to a specific entry number
   - Only entries with "split_sentence" label are shown

5. PROCESSING:
   - "Apply & Next" - Apply splits and move to next entry
   - "Skip" - Keep original entry and move to next
   - "Clear Splits" - Remove all split points for current entry
   - "Process All Remaining" - Auto-process remaining entries

6. AUTO-SAVE FEATURES:
   - Progress is automatically saved every 50 entries (configurable)
   - Progress is saved when you close the application
   - Manual save available via Ctrl+P or File menu
   - Session can be resumed if interrupted

7. SAVING:
   - "Save Processed" (Ctrl+S) - Save only the processed split_sentence entries
   - "Save All" (Ctrl+Shift+S) - Save all entries (original + processed)
   - After successful save, you can choose to clear the progress file
   
8. VIEWING:
   - "Show Statistics" - View file and processing statistics
   - "Show All Entries" - View all entries with processing status
   - "Show Progress Details" - View detailed progress information

KEYBOARD SHORTCUTS:
   Ctrl+O - Open file
   Ctrl+S - Save processed entries
   Ctrl+Shift+S - Save all entries
   Ctrl+P - Save progress manually
   Ctrl+C - Clear splits
   Ctrl+D - Auto-detect sentences
   Ctrl+G - Go to entry
   Ctrl+Q - Quit (with save prompt)
   Left/Right arrows - Navigate entries
   Enter - Apply & Next

NOTES:
   - Only entries with "split_sentence" label are displayed for editing
   - All other entries are preserved in the "Save All" option
   - Empty segments are automatically filtered out
   - Original entry fields are preserved in split entries
   - Progress is tracked per original entry, not per split
        """
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Instructions")
        dialog.geometry("700x650")
        dialog.transient(self.root)
        
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, width=80, height=40)
        text.pack(padx=10, pady=10)
        text.insert(tk.END, instructions)
        text.config(state=tk.DISABLED)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """JSONL Sentence Splitter
Version 3.0 - With Auto-Save

A GUI tool for splitting medical observation sentences
in JSONL files at user-specified positions.

NEW IN VERSION 3.0:
• Automatic progress saving every N entries
• Save progress on application exit
• Resume interrupted sessions
• Manual progress save (Ctrl+P)
• Progress tracking and statistics
• Settings dialog for customization

This version processes ONLY entries marked with 
the "split_sentence" label, making it efficient for
targeted text processing workflows.

Features:
• Filtered view of split_sentence entries only
• Interactive click-to-split interface
• Automatic sentence boundary detection
• Preview of split results
• Auto-save progress with session restoration
• Preserves all original data fields
• Saves both processed and unprocessed entries

Created for processing medical imaging observations."""
        
        messagebox.showinfo("About", about_text)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = JSONLSplitterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
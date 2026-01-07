"""
Backblaze Status GUI
A simple GUI application to monitor Backblaze backup status
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from datetime import datetime
from backblaze_client import BackblazeClient


class BackblazeStatusGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Backblaze Backup Status Monitor")
        self.root.geometry("900x700")

        self.bz_client = BackblazeClient()
        self.refresh_interval = 10000  # 10 seconds

        self.setup_ui()
        self.update_status()

    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Status Section
        status_frame = ttk.LabelFrame(main_frame, text="Current Status", padding="10")
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)

        # Status labels
        ttk.Label(status_frame, text="Client Status:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.status_label = ttk.Label(status_frame, text="Unknown", font=('Arial', 10, 'bold'))
        self.status_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(status_frame, text="Current File:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.current_file_label = ttk.Label(status_frame, text="None", foreground="blue")
        self.current_file_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(status_frame, text="Progress:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.progress_label = ttk.Label(status_frame, text="0%")
        self.progress_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(status_frame, text="Total Pending:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.pending_label = ttk.Label(status_frame, text="0 files")
        self.pending_label.grid(row=3, column=1, sticky=tk.W, pady=2)

        ttk.Label(status_frame, text="Last Updated:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.last_update_label = ttk.Label(status_frame, text="Never")
        self.last_update_label.grid(row=4, column=1, sticky=tk.W, pady=2)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.refresh_button = ttk.Button(button_frame, text="Refresh Now", command=self.manual_refresh)
        self.refresh_button.pack(side=tk.LEFT, padx=5)

        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Auto-refresh",
                       variable=self.auto_refresh_var).pack(side=tk.LEFT, padx=5)

        # Files list section
        files_frame = ttk.LabelFrame(main_frame, text="Files Scheduled for Backup", padding="10")
        files_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        # Create Treeview for files
        columns = ('filename', 'size', 'status')
        self.tree = ttk.Treeview(files_frame, columns=columns, show='headings', height=15)

        self.tree.heading('filename', text='File Path')
        self.tree.heading('size', text='Size')
        self.tree.heading('status', text='Status')

        self.tree.column('filename', width=500)
        self.tree.column('size', width=100)
        self.tree.column('status', width=150)

        # Scrollbars
        vsb = ttk.Scrollbar(files_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(files_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

    def format_size(self, size_bytes):
        """Format file size in human-readable format"""
        if size_bytes is None:
            return "Unknown"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def update_status(self):
        """Update the status display"""
        if not self.auto_refresh_var.get():
            # Schedule next update even if not refreshing
            self.root.after(self.refresh_interval, self.update_status)
            return

        self.status_bar.config(text="Updating...")
        self.refresh_button.config(state='disabled')

        # Run update in separate thread to avoid blocking UI
        thread = threading.Thread(target=self.fetch_and_update, daemon=True)
        thread.start()

        # Set a watchdog timer - if update doesn't complete in 30 seconds, re-enable button
        self.root.after(30000, self.check_update_timeout)

    def check_update_timeout(self):
        """Check if an update is taking too long"""
        if self.refresh_button['state'] == 'disabled':
            self.refresh_button.config(state='normal')
            self.status_bar.config(text="Update timed out - click Refresh to try again")

    def fetch_and_update(self):
        """Fetch data from Backblaze and update UI"""
        try:
            # Get status from Backblaze client
            status_info = self.bz_client.get_status()
            pending_files = self.bz_client.get_pending_files()

            # Schedule UI update on main thread
            self.root.after(0, lambda: self.update_ui(status_info, pending_files))

        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))

    def update_ui(self, status_info, pending_files):
        """Update UI elements with fetched data"""
        try:
            # Update status labels
            client_status = status_info.get('status', 'Unknown')
            self.status_label.config(text=client_status)

            # Color code the status
            if client_status.lower() in ['running', 'backing up']:
                self.status_label.config(foreground='green')
            elif client_status.lower() == 'paused':
                self.status_label.config(foreground='orange')
            else:
                self.status_label.config(foreground='red')

            # Update current file
            current_file = status_info.get('current_file', 'None')
            self.current_file_label.config(text=current_file if current_file else 'None')

            # Update progress
            progress = status_info.get('progress', 0)
            self.progress_label.config(text=f"{progress}%")

            # Update pending count
            self.pending_label.config(text=f"{len(pending_files)} files")

            # Update last updated time
            self.last_update_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Clear existing items in tree
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Add pending files to tree
            if pending_files:
                for file_info in pending_files:
                    filename = file_info.get('path', 'Unknown')
                    size = self.format_size(file_info.get('size', 0))
                    file_status = file_info.get('status', 'Pending')

                    self.tree.insert('', tk.END, values=(filename, size, file_status))

                self.status_bar.config(text=f"Updated successfully at {datetime.now().strftime('%H:%M:%S')} - {len(pending_files)} file(s) found")
            else:
                # Show message when no files found
                self.tree.insert('', tk.END, values=('No pending backup files found or unable to access Backblaze data', '-', 'N/A'))
                self.status_bar.config(text=f"Updated at {datetime.now().strftime('%H:%M:%S')} - No pending files detected")

        except Exception as e:
            self.show_error(f"Error updating UI: {str(e)}")
        finally:
            self.refresh_button.config(state='normal')
            # Schedule next update
            self.root.after(self.refresh_interval, self.update_status)

    def manual_refresh(self):
        """Manually trigger a refresh"""
        self.update_status()

    def show_error(self, error_msg):
        """Show error message"""
        self.status_bar.config(text=f"Error: {error_msg}")
        self.refresh_button.config(state='normal')
        # Schedule next update
        self.root.after(self.refresh_interval, self.update_status)


def main():
    root = tk.Tk()
    app = BackblazeStatusGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

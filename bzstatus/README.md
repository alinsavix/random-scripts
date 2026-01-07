# Backblaze Status Monitor

A simple Python GUI application to monitor the status of your Backblaze backup client on Windows.

## Features

- **Real-time Status Monitoring**: View the current state of your Backblaze client (Running, Paused, etc.)
- **Current File Display**: See which file is currently being backed up
- **Pending Files List**: View all files scheduled for backup with their sizes
- **Progress Tracking**: Monitor backup progress percentage
- **Auto-refresh**: Automatically updates every 10 seconds
- **Clean GUI**: Simple and intuitive tkinter-based interface

## Requirements

- Python 3.6 or higher
- Windows OS with Backblaze installed
- No external Python packages required (uses only standard library)

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd bzstatus
```

2. No additional dependencies needed - the application uses only Python standard library modules.

## Usage

Run the application with:

```bash
python bzstatus_gui.py
```

Or on Windows, you can double-click `bzstatus_gui.py` if Python is associated with `.py` files.

## How It Works

The application monitors Backblaze by:

1. **Service Detection**: Checks if the Backblaze Backup Service is running
2. **CLI Interface**: Attempts to use Backblaze executable to retrieve file lists
3. **Database Parsing**: Reads Backblaze database files for file information
4. **Log Analysis**: Parses log files to detect files being backed up
5. **File Filtering**: Excludes Backblaze's own internal files from the display

### Data Sources

The application attempts to read from several Backblaze data locations:
- `C:\ProgramData\Backblaze\bzdata\bzbackup\` - Main data directory
- Database files (`bzfilelist.dat`, `bzfileids.dat`)
- Status XML files
- Log files (`bz_done.txt`, `bztransmit.log`, etc.)
- TODO/pending file lists

**Important**: The application filters out Backblaze's own data files (`.dat`, `.xml`, etc. in `ProgramData\Backblaze`) and only shows actual user files that are scheduled for backup.

## GUI Overview

### Status Section
- **Client Status**: Shows if Backblaze is Running, Paused, or Not Running (color-coded)
- **Current File**: Displays the file currently being backed up
- **Progress**: Backup progress percentage
- **Total Pending**: Number of files waiting to be backed up
- **Last Updated**: Timestamp of last status refresh

### Control Buttons
- **Refresh Now**: Manually trigger a status update
- **Auto-refresh**: Toggle automatic updates (every 10 seconds)

### Files List
A scrollable table showing:
- Full file paths of pending files
- File sizes in human-readable format (KB, MB, GB)
- Status of each file

## Debugging

If you're having issues with the application not finding files, run the debug script:

```bash
python debug_backblaze.py
```

This will show you:
- Backblaze installation locations
- Available data directories and files
- Service status
- What files the application can access

## Troubleshooting

### Application shows "Unknown" status
- Ensure Backblaze is installed and running
- Check that the Backblaze data directory exists at `C:\ProgramData\Backblaze`
- Run the application as Administrator if permission issues occur

### No pending files shown
- If you only see Backblaze internal files, the app now filters those out
- The application tries multiple methods to find user files scheduled for backup
- Run `debug_backblaze.py` to see what Backblaze files are available on your system
- Check Backblaze permissions
- Verify Backblaze is actively backing up files
- The pending file list depends on Backblaze's internal tracking - if all files are backed up, the list may be empty

### Status not updating
- Ensure "Auto-refresh" checkbox is enabled
- Click "Refresh Now" to manually update
- Check that Backblaze service is running

## Limitations

- **Windows Only**: Currently designed for Windows systems
- **Read-Only**: This is a monitoring tool only - it cannot control Backblaze
- **File Access**: Requires read access to Backblaze data directories
- **Accuracy**: Status information depends on Backblaze's internal file structure, which may vary by version

## Creating a Standalone Executable

To create a standalone `.exe` file that doesn't require Python installation:

1. Install PyInstaller:
```bash
pip install pyinstaller
```

2. Create the executable:
```bash
pyinstaller --onefile --windowed --name "Backblaze Status Monitor" bzstatus_gui.py
```

3. Find the executable in the `dist` folder

## Future Enhancements

Potential features for future versions:
- Support for macOS and Linux
- Ability to pause/resume backups
- Historical backup statistics
- System tray integration
- Notifications for completed backups
- Detailed backup logs viewer

## License

This project is provided as-is for personal use.

## Contributing

Contributions, issues, and feature requests are welcome!

## Author

Created for monitoring Backblaze backup status

## Disclaimer

This is an unofficial tool and is not affiliated with or endorsed by Backblaze, Inc. Use at your own risk.

MAPLEBOT - READ ME FIRST
========================

Install
  1. Right-click the zip, choose Properties, tick Unblock, then OK.
  2. Extract it into %LOCALAPPDATA%\Programs\. You get %LOCALAPPDATA%\Programs\Maplebot\.
     If your Windows user name has non-English letters, extract it to C:\ instead (C:\Maplebot).
     Do not use Program Files or a OneDrive folder.
  3. Python, PowerShell and Claude are not needed.

Run
  1. Double-click Maplebot.exe. The settings app opens without an administrator prompt.
  2. First time: click Browse... next to Character image and pick your nametag crop, select the
     monsters, then Save & apply. Use Calibrate minimap and the Map editor to record your map.
     Or click "Load example settings..." to start from the included Construction Site example,
     if this build has it. If you already use Maplebot in another folder on this PC, click
     "Import settings..." and choose that folder instead (see below).
  3. Bring MapleStory to the front and click Start active. Windows asks for administrator
     approval every time; approve it to let Maplebot send keys. If you decline, nothing starts.
  4. Pause all, Hold farming, Return to point 1 and Stop work as before. Hold farming keeps HP
     protection active; Pause all and Stop do not.
  If Start says another controller is running, stop that one first. This includes a Maplebot
  started from a source checkout.

Bringing your settings from another Maplebot folder (this PC only)
  Stop every Maplebot controller first (the other folder's one too). Then either click
  "Import settings..." in the settings app, or close the settings app and run:
      maplebot-cli.exe import-settings "D:\path\to\other\maplebot"
  (If the settings app was left open, its next Save only reloads the imported settings for you
  to review; it does not save its old values over them.)
  It copies that folder's config\user.yaml and config\map_profiles\ into this folder. It only
  works while this folder's settings are still empty (add --force to replace them; replaced
  files are kept as *.before-import-* backups). Nothing is deleted. Image files named in the
  settings (for example your nametag crop) must also be in this folder's assets\; the import
  warns about any that are missing.

Command line (optional, maplebot-cli.exe in the same folder)
  maplebot-cli.exe status                 whether a controller is running
  maplebot-cli.exe stop                   asks the controller to stop and release its keys
  maplebot-cli.exe start [--wait-for-start]   starts the controller (administrator prompt)
  maplebot-cli.exe supervise --output session.jsonl --minutes 60   optional session supervisor
  maplebot-cli.exe selftest               offline check of this installation (sends no input)
  maplebot-cli.exe paths                  where the program and your data live
  maplebot-cli.exe restore-example        same as "Load example settings..."
  maplebot-cli.exe import-settings <folder> [--force]
  maplebot-cli.exe version

Your data
  config\user.yaml, config\map_profiles\ and log\ stay in this folder. They are never part of
  a Maplebot download.

Upgrade
  1. Click Stop and wait for Stopped. Stop any `maplebot-cli supervise`. Close the settings app.
  2. In Task Manager, check that no Maplebot.exe or maplebot-cli.exe is running.
  3. Delete Maplebot.exe, maplebot-cli.exe and _internal.
  4. Unblock and extract the new zip into the same parent folder, %LOCALAPPDATA%\Programs\.
  Your config\user.yaml, config\map_profiles\ and log\ are kept. Local edits to shipped files
  are replaced, so put your overrides in config\user.yaml.

Important
  Run only one Maplebot at a time. Do not put .py files or Python folders into the Maplebot
  folder; the controller refuses to start if it finds them. Antivirus may warn about an
  unsigned program that sends keys: check SHA256SUMS.txt, and do not add broad exclusions.
  Never share a used Maplebot folder: it contains your settings and logs. Share only the zip.

@echo off
rem Copyright (c) 2022-2026, The Isaac Lab Project Developers.
rem SPDX-License-Identifier: BSD-3-Clause
wsl.exe -d Ubuntu -- bash /home/nmt/Projects/Text2HOI/_local/text2hoi/run_mug.sh
if errorlevel 1 (
    echo Text2HOI failed. See the error above.
) else (
    echo Finished. Results are in \\wsl.localhost\Ubuntu\home\nmt\Projects\Text2HOI\_local\text2hoi\outputs
)
pause

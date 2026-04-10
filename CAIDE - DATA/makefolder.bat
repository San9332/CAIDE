@echo off
for /L %%i in (1,1,75) do (
    set num=00%%i
    call mkdir P%%num:~-2%%
)
pause
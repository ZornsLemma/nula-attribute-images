REM Code based on tricky's game.asm, but rather hacked about
pause%=500:REM centiseconds
pal=&3000-&800
init_nula_pal=pal-32
init_ula_pal=init_nula_pal-16
*LOAD CODE 900
*TV255,1
MODE 1
VDU 23;8202;0;0;0;
*FX229,1
HIMEM=&2700
?&FE22=&61
FOR I%=0 TO 15
init_ula_pal?I%=(I%*16)+(I% EOR 7)
NEXT
REPEAT
CLS
READ F$
IF F$="XXX" THEN RESTORE:READ F$
OSCLI "LOAD "+F$+ " 27D0"
REMPROCstripes
!&80=&10000-(pause%/2)
CALL &900
UNTIL FALSE
END
FOR I%=0 TO 15
J%=I%
IF I%=3 THEN J%=15
IF I%=15 THEN J%=3
init_nula_pal?(I%*2)=J%+J%*16
init_nula_pal?(I%*2+1)=I%*16+J%
NEXT
FOR Z%=pal TO pal+&7FF
?Z%=&00
NEXT
FOR I%=1 TO 255 STEP 3
REM 3->red
pal?(I%+&000)=&3F:pal?(I%+&100)=&00
REM 3->green
pal?(I%+&001)=&00:pal?(I%+&101)=&00
pal?(I%+&201)=&30:pal?(I%+&301)=&F0
REM 3->blue
pal?(I%+&202)=&00:pal?(I%+&302)=&00
pal?(I%+&402)=&30:pal?(I%+&502)=&0F
NEXT
FOR I%=&3000 TO &7FFC STEP 4
!I%=&EEEEEEEE
NEXT
CALL start
DEF PROCstripes
FOR Y%=0 TO 31
S%=&3000+Y%*640+30*8
FOR X%=0 TO 15
FOR Y2%=0 TO 7
A%=X% MOD 4
IF A%=0 THEN ?S%=&0
IF A%=1 THEN ?S%=8+4+2
IF A%=2 THEN ?S%=128+64+32
IF A%=3 THEN ?S%=128+64+32+8+4+2
B%=X% DIV 4
?S%=(?S%)+((B% AND 2)*8)+(B% AND 1)
S%=S%+1
NEXT
NEXT
NEXT
ENDPROC
DATA JAFFA
DATA PARROT
DATA CAR
DATA TOWER
DATA BEACH
DATA ROCKS
DATA PLAZA
DATA DESERT
DATA XXX

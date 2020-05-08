REM Code based on tricky's game.asm, but rather hacked about
pause%=500:REM centiseconds
sm_im=1
IntCA1=2
SysIntVSync=IntCA1
SystemVIA=&FE40
ViaIFR=13
SysViaIFR=SystemVIA+ViaIFR
CrtcReg=&FE00
CrtcVal=&FE01
CrtcR1HorizontalDisplayed=1
CrtcR2HorizontalSyncPosition=2
CrtcR3SyncPulseWidths=3
ulapal=&FE21
nulapal=&FE23
pal=&3000-&800
init_nula_pal=pal-32
init_ula_pal=init_nula_pal-16
nulaotf%=FALSE:REM will vary between images
IF nulaotf% THEN updatepal=nulapal ELSE updatepal=ulapal
FOR opt%=0 TO 2 STEP 2
P%=&900:REM we need to avoid page crossing so don't DIM code space
[OPT opt%
.start
        sei

\ TODO We don't really need this as all our images are full width, though probably as 
\ well to keep it since tricky has it, and the sync pulse width change may help
\ real hardware
        ldx #CrtcR1HorizontalDisplayed     : stx CrtcReg
        lda #&50                      : sta CrtcVal
        ldx #CrtcR2HorizontalSyncPosition  : stx CrtcReg
         lsr A : adc #98-40 : sta CrtcVal
        ldx #CrtcR3SyncPulseWidths         : stx CrtcReg
        lda #&29                           : sta CrtcVal ; because my LCD doesn't sync with &28!

]
REM One-off initialisation of whichever palette isn't changing on the fly
IF nulaotf% THEN [OPT FNinit_ula_pal:] ELSE [OPT FNinit_nula_pal:]
[OPT opt%
.loop

\ Wait for VSYNC
        lda #SysIntVSync
        sta SysViaIFR
.wait_vsync
        bit SysViaIFR
        beq wait_vsync

\ Load the initial palette to start the frame
\ TODO I haven't cycle matched tricky's code here, do I need to? probably...
]
IF nulaotf% THEN [OPT FNinit_nula_pal:] ELSE [OPT FNinit_ula_pal:]
REM TODO: No obvious reason y0x should differ any more...
IF nulaotf% THEN y0x=237 ELSE y0x=234
[OPT opt%
\ Wait out line Y=0
        ldx #y0x
.waitt
        pha : pla : pha : pla
        dex
        bne waitt
        pha : pla
\       beq *+2
        nop

\ Do lines Y=1-255
        ldy #1
.next_line

        \ Get ready for the time-critical bit to come.
        \ 60 cycles in this block
        lda pal+&000,y:sta update0+sm_im \ 8 cycles
        lda pal+&100,y:sta update1+sm_im \ 8 cycles
        lda pal+&200,y:sta update2+sm_im \ 8 cycles
        lda pal+&300,y:sta update3+sm_im \ 8 cycles
        lda pal+&400,y:sta update4+sm_im \ 8 cycles
        lda pal+&500,y:sta update5+sm_im \ 8 cycles
        lda pal+&600,y:sta update6+sm_im \ 8 cycles
        ldx pal+&700,y                   \ 4 cycles

        \ These updates have to occur within the 48 cycles of the horizontal blanking
        \ interval to avoid twinkling. We have 46 cycles here, and the first two are
        \ not actually touching hardware so we do all the critical stuff in 44 cycles.
.update0
        lda #0:sta updatepal \ 6 cycles
.update1
        lda #0:sta updatepal \ 6 cycles
.update2
        lda #0:sta updatepal \ 6 cycles
.update3
        lda #0:sta updatepal \ 6 cycles
.update4
        lda #0:sta updatepal \ 6 cycles
.update5
        lda #0:sta updatepal \ 6 cycles
.update6
        lda #0:sta updatepal \ 6 cycles
.update7
        stx updatepal        \ 4 cycles

        \ We need to take 128 cycles per line. We've had 60+46 so far and the
        \ code at foo below will take 5 cycles, so we need to burn 128-60-46-5=17
        \ cycles.
        nop     \ 2 cycles
        nop     \ 2 cycles
        nop     \ 2 cycles
        nop     \ 2 cycles
        nop     \ 2 cycles
        nop     \ 2 cycles
        nop     \ 2 cycles
        and &70 \ 3 cycles
.foo

        iny : bne next_line \ 5 cycles if taken assuming no page crossing
        inc &80:beq maybe
        jmp loop
.maybe
        inc &81:beq done
        jmp loop
.done
        rts

]
NEXT
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
F$="TOWER"
OSCLI "LOAD "+F$+ " 27D0"
REMPROCstripes
REPEAT
!&80=0:REM!&80=&10000-(pause%/2)
CALL start
UNTIL FALSE
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
DEF FNinit_ula_pal
[OPT opt%
        \ X=16-31 is garbage data, but we'll program with correct data afterwards
        \ and by doing it this way the timing is the same as FNinit_nula_pal.
        ldx #31
.init_ula_palette
        lda init_ula_pal,x
        sta ulapal
        dex
        bpl init_ula_palette
]
=opt%
DEF FNinit_nula_pal
[OPT opt%
        ldx #31
.init_col
        lda init_nula_pal,x
        sta nulapal
        dex
        bpl init_col
]
=opt%
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
DATA PIZZA
DATA TOWER
DATA BEACH
DATA ROCKS
DATA DESERT
DATA XXX

REM Code based on tricky's game.asm, but rather hacked about
HIMEM=&2700
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
FOR opt%=0 TO 3 STEP 3
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
IF nulaotf% THEN y0x=237 ELSE y0x=238
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

        \ tricky's code was for ULA palette, so we have to tweak this bit.
        \ His code with cycle counts (assuming no page crossing)
        \ 8=4+4 lda cols+&000,y ; sta col0+sm_im
        \ 8=ditto lda cols+&100,y ; sta col1+sm_im
        \ 24=2+4+2+4+2+4+2+4 .col0 lda #5 ; STA VideoULAPalette ; EOR #&10 ; STA VideoULAPalette ; EOR #&40 ; STA VideoULAPalette ; EOR #&10 ; STA VideoULAPalette
        \ 24=ditto .col1 lda #3 ; STA VideoULAPalette ; EOR #&10 ; STA VideoULAPalette ; EOR #&40 ; STA VideoULAPalette ; EOR #&10 ; STA VideoULAPalette
        \ A total of 64 cycles

        \ TODO See tricky's post in "my" stardot thread, he has some advice which may make 8 possible
        lda #0:sta&82
        lda pal+&000,y:sta updatepal \ 8 cycles
        and #&F0:cmp #&00:beq boom1
        lda pal+&100,y:sta updatepal \ 8 cycles
        and #&F0:cmp #&00:beq boom2
        lda pal+&200,y:sta updatepal \ 8 cycles
        and #&F0:cmp #&00:beq boom3
        lda pal+&300,y:sta updatepal \ 8 cycles
        and #&F0:cmp #&00:beq boom4
        lda pal+&400,y:sta updatepal \ 8 cycles
        and #&F0:cmp #&00:beq boom5
        lda pal+&500,y:sta updatepal \ 8 cycles
        and #&F0:cmp #&00:beq boom6
        lda pal+&600,y:sta updatepal \ 8 cycles
        and #&F0:cmp #&00:beq boom7
        lda pal+&700,y:sta updatepal \ 8 cycles
        and #&F0:cmp #&00:beq boom8
        \ A total of 64 cycles, same as tricky's ULA palette code

        \ Following code up to and including foo takes
        \ 2+10*(2+3)+1*(2+2)+3=59 cycles
        ldx #11 \ 2 cycles
.wait
        dex \ 2 cycles
        bne wait \ 2 cycles if not taken, 3 cycles if taken assuming no page crossing
        beq foo \ 3 cycles
.foo

        iny : bne next_line \ 5 cycles if taken assuming no page crossing
        \ So if branch is taken, we have burned 64+59+5=128 cycles

\ HACK
        \dec &80
        \beq foo2
        jmp loop
.foo2
.boom1
        inc &82
.boom2
        inc &82
.boom3
        inc &82
.boom4
        inc &82
.boom5
        inc &82
.boom6
        inc &82
.boom7
        inc &82
.boom8
        inc &82
        sty &81
        rts

]
NEXT
?&80=255:REM HACK
*TV255,1
MODE 1
VDU 23;8202;0;0;0;
?&FE22=&61
FOR I%=0 TO 15
init_ula_pal?I%=(I%*16)+(I% EOR 7)
NEXT
*LOAD JAFFA 27D0
PROCstripes
CALL start
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

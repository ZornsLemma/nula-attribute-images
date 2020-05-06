REM Code based on tricky's game.asm, but rather hacked about
IntCA1=2
SysIntVSync=IntCA1
SystemVIA=&FE40
ViaIFR=13
SysViaIFR=SystemVIA+ViaIFR
nulapal=&FE23
pal=&3000-&800
FOR opt%=0 TO 3 STEP 3
P%=&900:REM we need to avoid page crossing so don't DIM code space
[OPT opt%
.start
	sei

\ Load the initial palette
\ TODO I haven't cycle matched tricky's code here, do I need to? probably...
	ldx #31
.init_col
	lda init_pal,x
	sta nulapal
	dex
	bpl init_col

.loop

\ Wait for VSYNC
	lda #SysIntVSync
	sta SysViaIFR
.wait_vsync
	bit SysViaIFR
	beq wait_vsync

\ tricky's code initialises the ULA palette here. TODO For now I've left a
\ no-op version if it here to preserve cycle counts, fiddle with this later
        lda dummy
        ldy #7
        ldx #3
.init_col1
        asl A
        bcc unused
        pha
        tya
        ora init_pal,x
        STA &8000 : EOR #&10 : STA &8000 : EOR #&40 : STA &8000 : EOR #&10 : STA &8000
        dex
        pla
.unused
        dey
        bpl init_col1


\ Wait out line Y=0
	ldx #246
.waitt
	pha : pla : pha : pla
	dex
	bne waitt
	pha : pla
\	beq *+2
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

	lda pal+&000,y:sta nulapal \ 8 cycles
	lda pal+&100,y:sta nulapal \ 8 cycles
	lda pal+&200,y:sta nulapal \ 8 cycles
	lda pal+&300,y:sta nulapal \ 8 cycles
	lda pal+&400,y:sta nulapal \ 8 cycles
	lda pal+&500,y:sta nulapal \ 8 cycles
	lda pal+&600,y:sta nulapal \ 8 cycles
	lda pal+&700,y:sta nulapal \ 8 cycles
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

	jmp loop

.dummy
	equb &0f

.init_pal
	\ Note each pair is reversed, but equw handles that (this is only temporary though)
	equw &0000
	equw &1111
	equw &2222
	equw &3333
	equw &4444
	equw &5555
	equw &6666
	equw &7777
	equw &8888
	equw &9999
	equw &AAAA
	equw &BBBB
	equw &CCCC
	equw &DDDD
	equw &EEEE
	equw &FFFF
]
NEXT
MODE 1
VDU 23;8202;0;0;0;
?&FE22=&61
FOR I%=0 TO 15
?&FE21=(I%*16)+(I% EOR 7)
NEXT
FOR Z%=pal TO pal+&7FF
?Z%=&00
NEXT
FOR I%=1 TO 255 STEP 20
pal?(I%+&000)=&3F:pal?(I%+&100)=&00:REM 3->red
pal?(I%+&001)=&30:pal?(I%+&101)=&F0:REM 3->green
pal?(I%+&002)=&30:pal?(I%+&102)=&0F:REM 3->blue
NEXT
FOR I%=&3000 TO &4200
?I%=&EE
NEXT
CALL start

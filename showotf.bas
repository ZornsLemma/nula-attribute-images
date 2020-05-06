REM Code based on tricky's game.asm, but rather hacked about
IntCA1=2
SysIntVSync=IntCA1
SystemVIA=&FE40
ViaIFR=13
SysViaIFR=SystemVIA+ViaIFR
nulapal=&FE23
DIM code% 512
FOR opt%=0 TO 3 STEP 3
P%=code%
[OPT opt%
.loop

	lda #SysIntVSync
	sta SysViaIFR
.wait_vsync
	bit SysViaIFR
	beq wait_vsync

	ldx #31
.init_col
	lda init_pal,x
	sta nulapal
	dex
	bpl init_col
.sftodo jmp sftodo

	lda cols
	ldy #7
	ldx #3
.init_col
	asl A
	bcc unused
	pha
	tya
	ora init_pals,x
	STA VideoULAPalette : EOR #&10 : STA VideoULAPalette : EOR #&40 : STA VideoULAPalette : EOR #&10 : STA VideoULAPalette
	dex
	pla
.unused
	dey
	bpl init_col

	ldx #246
.waitt
	pha : pla : pha : pla
	dex
	bne waitt
	pha : pla
\	beq *+2
	nop

	ldy #1
.next_line

	lda cols+&000,y : sta col0+sm_im
	lda cols+&100,y : sta col1+sm_im

.col0 lda #5 : STA VideoULAPalette : EOR #&10 : STA VideoULAPalette : EOR #&40 : STA VideoULAPalette : EOR #&10 : STA VideoULAPalette
.col1 lda #3 : STA VideoULAPalette : EOR #&10 : STA VideoULAPalette : EOR #&40 : STA VideoULAPalette : EOR #&10 : STA VideoULAPalette

	ldx #11
.wait
	dex
	bne wait
	beq foo
.foo

	iny : bne next_line
	rts

.init_pal
	\ Note each pair is reversed
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
FOR I%=&3000 TO &7FFF
?I%=&FC
NEXT
CALL code%

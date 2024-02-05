.section .text
.global _entry
_entry:
        la 		sp, stack0
        call    start

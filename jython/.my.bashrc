###########
#
# User customization file for bash
#
############

umask 002

# Prepend user's bin directory to path
PATH=$HOME/bin:$PATH

# Only set these up for interactive shells
# For example, any 'stty' calls should be inside the if/fi
if [ "${PS1:+set}" = set ]; then
	export EDITOR=vim
	export VISUAL=$EDITOR
	export EXINIT="set ai aw sm"
	export FCINIT emacs
	export PAGER=less
	export LESS=-ce
	export MAIL=/usr/spool/mail/$USER
	export MAILCHECK=30
	export MAILFILE=$MAIL
	export PRINTER=lw

	alias	l='ls -lg -h -a --color=auto'
	alias	ls='ls -CF'
	alias	ll='ls -lg'
	alias	la='ls -A'
	alias	lla='ls -Alg'
	alias	passwd=yppasswd
	alias	sun='stty dec; stty erase \^H'
	alias	dec='stty dec'
	alias	tree='~/tree --dirsfirst'
	alias	treed='~/tree -d --dirsfirst'
	alias	treedu='~/tree --du -h --dirsfirst'
	alias	xtitle='echo -n "]0;\!*"'

	PS1='\[\033[0;33m\]\u@\h \[\033[1;31m\]\w\[\033[${?/[^0]/39}m\]\$ \[\033[0;38m'

	fqc="/home/unix/feldman/FastQC/fastqc"
	alias less="less -S"
	alias kol="column -s, -t"
	alias fiji="/broad/blainey_lab/David/software/Fiji.app/ImageJ-linux64"
	cd /broad/blainey_lab/David
	set bell-style none
	bind 'TAB: menu-complete'
	alias ~!='cd /broad/blainey_lab/David'
	alias hist="sort | uniq -c | sort -r"
	alias tar='tar -zxvf'

fi

# Load necessary modules/software
use .matlab-2013a
#use .rstudio-0.97.revolutionr_6.0.0
use Bowtie
use Samtools
use BamTools
use VCFtools
use BEDTools
use .homer-4.1
#use .meme-4.9.1
use .r-2.15.3
use Git-1.7
use .anaconda-2.1.0-no-mkl
use .ghostscript-9.10
use BWA
use FASTX-Toolkit 
use ViennaRNA
use Java-1.8
use UGER

# ViennaRNA commands and utilities, not automatically added...
PATH=/broad/software/free/Linux/redhat_5_x86_64/pkgs/viennarna_2.1.5/bin:$PATH
PATH=/broad/software/free/Linux/redhat_5_x86_64/pkgs/viennarna_2.1.5/share/ViennaRNA/bin:$PATH
PATH=/broad/blainey_lab/David/packages/lasagna:$PATH

setenv BLASTDB "/broad/data/blastdb/nt"

#for n-grams stuff

#export PATH=${PATH}:~/ngram

#####

# Setting up EPD7.1-2 FOR PYTHON ONE HAS TO USE THIS ODD FORMAT OF PATH SET UP
#PATH="/home/unix/dfernand/bin/epd-7.1-2-rh5-x86_64/bin:/home/unix/dfernand/bin/pythonmodules:$PATH"
#export PATH
# Set Python Path EPD7.2 Distro.
#PYTHONPATH="/home/unix/dfernand/bin/lib/python2.7/site-packages/:/home/unix/dfernand/bin/epd-7.1-2-rh5-x86_64/lib/python2.7/site-packages/:$PYTHONPATH"
#export PYTHONPATH

alias bsub_='bsub -R"rusage[mem=4000]" -q bhour -o deleteme -P lasagna'
alias bsub_10='bsub -R"rusage[mem=10000]" -q bhour -o deleteme -P lasagna'

#alias for submitting an interactive LSF job
alias bigish20='bsub -R"rusage[mem=20000]" -q iweek -W 240 -Is /bin/bash -l'
alias bigish10='bsub -R"rusage[mem=10000]" -q iweek -W 240 -Is /bin/bash -l'
alias bigish5='bsub -R"rusage[mem=5000]" -q iweek -W 240 -Is /bin/bash -l'

alias ish20='bsub -Is -q interactive -R "rusage[mem=20000]" "bash"'
alias ish10='bsub -Is -q interactive -R "rusage[mem=10000]" "bash"'

setenv TERM xterm-color




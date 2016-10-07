#!/bin/bash
HOST="ftp.cluster003.hosting.ovh.net"
USER="***"
PASS="***"
FTPURL="ftp://$USER:$PASS@$HOST"
LCD="_site"
RCD="/www"
#DELETE="--delete"
lftp -c "set ftp:list-options -a;
open '$FTPURL';
lcd $LCD;
cd $RCD;
mirror --reverse \
           $DELETE \
           --verbose \
           --exclude-glob a-dir-to-exclude/ \
           --exclude-glob a-file-to-exclude \
           --exclude-glob a-file-group-to-exclude* \
           --exclude-glob other-files-to-exclude"

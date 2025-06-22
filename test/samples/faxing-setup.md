   

# Faxing Setup

This function allows you to set up faxing for incoming and outgoing faxes. You
can set COM port settings, *Brooktrout* settings, and dialing rules.

The Integrated Faxing Application is already a part of Applied Epic; however, a
license is required to activate the application. This is activated by installing
the Fax Server Client (FSC) on the dedicated fax server. Click here for
installation instructions.

Once you have installed the Fax Server Client, use the following instructions to
configure your faxing.

1. From the Home screen, do one of the following:  
    
   * Click **Configure** on the navigation panel.
   * Click the **down arrow** next to *Home* on the menubar and select **Configure**.
   * Click **Areas > Configure** on the menubar.

From any other area of the program, do one of the following:

2. * Click the **down arrow** to the right of the *Home* options bar button and select **Configure**.
   * Click **Home > Configure** on the menubar. The *Configure* screen displays.

2. Click **Job Management** on the navigation panel, or **Areas > Job Management** on the menubar.
3. Click **Fax Setup** on the navigation panel.
4. If this is your first time configuring faxing setup, the list at the top of the screen is blank. Click the **Add** button to add a Fax Server Client.
5. You are presented with a list of workstations in your system with the Fax Server Client installed. Because only one Fax Server Client can be associated per database, in many cases you will only run it on one station. Check the appropriate **workstation(s)** and click **Finish**.
6. The *Fax Server Client IP Address* you just added displays in the list with a *Status* of *Pending* (meaning connection is still being established). Click the **Refresh** button at the top of the list to monitor the current status.
7. When the *Status* changes to *Attached*, the Fax Server Client is available to be configured. Highlight the **Fax Server Client** in the list and click the **Edit** button.
8. Fill out the following fields beneath the list:  
   * **Fax server client name**- Enter the computer name running the Fax Server Client or an alternate identifier for the workstation.
   * **Outgoing fax ID**- It is recommended to enter the phone number in use or an alternate identifier for the line.
   * **Retry count**- Enter the number of times you would like Applied Epic to reattempt sending a fax if a failure is encountered. If you do not enter a value of at least 1 in this field, no additional attempts will be made after a fax fails to send.
   * **Retry wait time (in seconds)**- Enter the number of seconds the Fax Server Client should wait after a failure before retrying. The recommended value is 180 (three minutes).
9. Fill out any applicable information on the four tabs below:  
   * COM Port Settings: If you are using *Brooktrout* channels only, you may skip this tab.

     1. The *COM Port Settings* list shows all communication ports available on the machine running the Fax Server Client. Check the **port(s)** that will be used.
     2. Highlight each selected port and select a **Port functionality** from the dropdown list:
        + **Send/Receive:** Select this option if you only have one communication port and must use it for both functions. If you have multiple ports, you may wish to designate one as *Receive Only* so that it will be available to receive faxes even while you are sending faxes.
        + **Send Only**
        + **Receive Only**
     3. Make the appropriate selections in the *Advanced COM Port Settings* frame:
        1. Determine whether you will enable *Error correction mode*. This transmission mode means that if the modem receives a fax containing bad data and cannot correct it, it will request that the sender return only the bad block of data, rather than the entire fax. Not all modems support ECM. Check your modem hardware before selecting one of the following:
           + **Enable ECM:** If you select this option and your modem hardware does not support ECM, you may experience errors when sending faxes.
           + **Disable ECM:** If ECM is disabled, the modem requests that the entire fax be resent when bad data is received.
           + **Allow ECM & Non-ECM:** If in doubt, select this option. This works for any setup, including a setup that supports ECM when sending faxes but not when receiving them.
        2. In the *Number of rings to wait before answering* field, enter **1** or **2**. You may wish to wait two rings if you have a complex phone system and want to ensure the fax has enough time to reach the modem.
        3. Optionally select **Hear modem speaker until connection is made**.
   * Brooktrout Settings: Applicable only if the Fax Server Client has a *Brooktrout* card installed.

     1. The *Brooktrout Channels* list shows all channels available on the machine running the Fax Server Client. Check the **channel(s)** that will be used.
     2. Highlight each selected channel and select a **Channel functionality** from the dropdown list:
        + **Send/Receive**
        + **Send Only**
        + **Receive Only**
     3. If your *Brooktrout* card supports Direct Inward Dialing, you can set it up in the *Routing for Direct Inward Dialing* frame. This feature enables you to route faxes directly to specific employees based on their Direct Inward Dialing numbers (the last four digits or their fax numbers).  
           
         **Note:** The *Brooktrout* channel used for Direct Inward Dialing must have a *Channel functionality* of *Receive Only*.
        1. Click the **Add** button to the left of the list. The *Add DID Routing* window displays.
        2. Check **All numbers** if all faxes should be routed to a single employee. If you are going to set up multiple routing recipients, leave *All numbers* unchecked and enter the last four digits of the first employee’s fax number in the *DID #* field.
        3. Select the corresponding **employee code** from the *Routing Recipient* dropdown list.
        4. To add another Direct Inward Dialing routing, click **Add**. When you are done adding routings, click **Finish**.
   * Area Code/Prefix Routing: Applicable only if you have multiple Fax Server Clients and would like to determine which faxes are sent from which Fax Server Client.

     1. Click the **Add** button to the left of the *Area Code* list to enter the area code you are faxing from.
     2. Enter the **code**. Click **Add** to enter any additional codes. When you are done adding area codes, click **Finish**.
     3. Highlight each **area code** and select a radio button in the *Prefixes* frame:
        + **Route all prefixes from the selected area code to the selected fax server client**
        + **Route selected prefixes from the selected area code to the selected fax server client:** If you select this option, check the appropriate **prefixes** in the list. You can also click *Select Range* to enter a range of prefixes to which this rule applies.
   * Dialing Options

     1. In the *General Dialing Rules* frame, enter your local area code in the *Area code dialing from* field.
     2. If you must dial any extra numbers in order to make a local call outside your organization, enter them in the *To access an outside line for local calls dial* field.
     3. If you must dial any extra numbers in order to make a long-distance call outside your organization, enter them in the *To access an outside line for long-distance calls dial* field.  
           
         **Note:** The Fax Server Client automatically dials 1 before long distance calls. Do **not** enter 1 in this field unless you must dial 1 twice to make long-distance calls. For example, if you need to dial 9 to access an outside line, but your phone system has no other special requirements, you should simply enter 9 for step b and step c.
     4. In the *International Dialing Rules* frame, enter your local country code in the *Country code dialing from* field.
     5. In the *International exit code* field, enter the code that must be used to fax internationally.
     6. To ensure your dialing rules are correct, click **Verify Dialing Options**. Enter a phone number (in ten-digit format, including area code) in the field provided and click **Continue**. The system displays the number as it would be dialed by the Fax Server Client, using the current dialing rules.
     7. If your phone system requires any special dialing rules for certain circumstances, you may enter them in the *Custom Dialing Rules* list. Click the **Add** button to enter a new rule.
        1. Enter the **Area code calling** to which this rule applies.
        2. Select the **Apply rules to all prefixes** or **Apply rules to selected prefixes** radio button. If you are applying the rule to selected prefixes only, highlight the appropriate **prefixes** in the list. You can press *[Ctrl]* while clicking to select multiple prefixes.
        3. Select the appropriate checkboxes:
           + **Dial \_\_\_ before:** Enter the correct number in the field.
           + **Include the area code**
           + **Dial \_\_\_ after:** Enter the correct number in the field.
        4. Click **Add** if you need to add another rule. When you are done adding rules, click **Finish**. You may also click **Cancel** to discard changes made.
   * Status Codes

     These status codes can assist you in troubleshooting any errors you may
     encounter.

     |  |  |
     | --- | --- |
     | Code | Description |
     | 0 | Sent successfully |
     | 1 | Session not terminated |
     | 2 | Unspecified error |
     | 3 | Ringback detected |
     | 4 | User abort |
     | 5 | No carrier detected |
     | 6 | Busy signal |
     | 7 | Unspecified phase A error |
     | 8 | Unspecified phase B error |
     | 9 | No answer |
     | 10 | No dial tone |
     | 11 | Remote modem cannot receive or send |
     | 12 | Cannot receive command |
     | 13 | Invalid command received |
     | 14 | Invalid response received |
     | 15 | DCS sent three times without answer |
     | 16 | DIS sent without answer |
     | 17 | DIS received three times |
     | 18 | DIS not received |
     | 19 | Failure training in 2400bps |
     | 20 | Failure training minimal send speed reached |
     | 21 | Error in page transmission |
     | 22 | Unspecified image format |
     | 23 | Image conversion error |
     | 24 | DTE to DCE data underflow |
     | 25 | Unrecognized transparent data command |
     | 26 | Image error line length wrong |
     | 27 | Page length wrong |
     | 28 | Wrong compression mode |
     | 29 | Unspecified phase D error |
     | 30 | No response to MPS |
     | 31 | Invalid response to MPS |
     | 32 | No response to EOM |
     | 33 | Invalid response to EOM |
     | 34 | No response to EOM |
     | 35 | Invalid response to EOM |
     | 36 | Unable to continue after PIN or PPP |
     | 37 | T.30 T2 timeout expected |
     | 38 | T.30 T1 timeout expected |
     | 39 | Missing EOL after 5 sec |
     | 40 | Bad CRC or frame |
     | 41 | DCE to DTE data underflow |
     | 42 | DCE to DTE data overflow |
     | 43 | Remote cannot receive |
     | 44 | Invalid ECM mode |
     | 45 | Invalid BFT(Binary File Transfer) mode |
     | 46 | Invalid width |
     | 47 | Invalid length |
     | 48 | Invalid compression |
     | 49 | Invalid resolution |
     | 50 | Remote cannot receive color faxes |
     | 51 | No transmitting document |
     | 52 | No response to PPS |
     | 53 | No modem or COM port |
     | 54 | Incompatible modem or COM port |
     | 55 | Unspecified Brooktrout error |
     | 56 | File I/O error |
     | 57 | Bad file format |
     | 58 | Firmware does not support capability |
     | 59 | Channel not in proper state (not connected) |
     | 60 | Bad parameter value used |
     | 61 | Memory allocation error |
     | 62 | Channel not in required state |
     | 63 | Brooktrout error: too soon |
     | 64 | No loop current detected |
     | 65 | Local phone in use |
     | 66 | Ringing detected during dialing |
     | 67 | Brooktrout error: no wink |
     | 68 | Confirmation tone |
     | 69 | The dialing sequence did not break the dial tone |
     | 70 | Group 2 fax machine detected |
     | 71 | Answer (probably human) detected |
     | 72 | No energy detected, possible dead line |
     | 73 | Recall dial tone detected |
     | 74 | Remote end did not answer |
     | 75 | Invalid number or class of service restriction |
     | 76 | No circuit detected, possible dead line |
     | 77 | Reorder tone detected |
     | 78 | Remote originating failure, invalid number |
     | 79 | RSPREC invalid response received |
     | 80 | DCN received in COMREC |
     | 81 | DCN received in RSPREC |
     | 82 | Incompatible fax formats |
     | 83 | Invalid DMA count specified for transmitter |
     | 84 | BFT specified, but ECM not enabled |
     | 85 | BFT specified, but not supported by receiver |
     | 86 | No response to RR after three tries |
     | 87 | No response to CTC or response not CTR |
     | 88 | T5 timeout since receiving first RNR |
     | 89 | Do not continue with next message after receiving ERR |
     | 90 | ERR response to EOR-EOP or EOR-PRI-EOP |
     | 91 | RSPREC error |
     | 92 | No response received after third try for EOR-NULL |
     | 93 | No response received after third try for EOR-MPS |
     | 94 | No response received after third try for EOR-EOP |
     | 95 | No response received after third try for EOR-EOM |
     | 96 | RSPREC error |
     | 97 | DCN received in COMREC |
     | 98 | DCN received is RSPREC |
     | 99 | Invalid DMA count specified for receiver |
     | 100 | BFT specified but ECM not supported by receiver |
     | 101 | RSPREC invalid response received |
     | 102 | COMREC invalid response received |
     | 103 | T3 timeout: no local response for voice interrupt |
     | 104 | T2 timeout: no command received after responding RNR |
     | 105 | DCN received for command received |
     | 106 | Command receive error |
     | 107 | Receive block count error in ECM mode |
     | 108 | Receive page count in ECM mode |
     | 109 | Human voice detected |
     | 500 | Server had problem creating image |
     | 600 | Unspecified error in transmission |
     | 601 | Ring detected without successful handshake |
     | 602 | Call aborted |
     | 603 | No loop current or A/B signaling bits |
     | 604 | ISDN disconnection |
     | 611 | No answer, T.30 T1 timeout |
     | 620 | Unspecified transmit Phase B error |
     | 621 | Remote cannot send or receive |
     | 622 | COMREC error, Phase B transmit |
     | 623 | COMREC invalid command received |
     | 624 | RSPREC error |
     | 625 | DCS sent three times without a response |
     | 626 | DIS/DTC received three times; DCS not recognized |
     | 627 | Failure to train |
     | 628 | RSPREC invalid response received |
     | 629 | DCN received in COMREC |
     | 630 | DCN received in RSPREC |
     | 633 | Incompatible fax formats (for example, a page width mismatch) |
     | 634 | Invalid DMA count specified for transmitter |
     | 635 | Binary File Transfer specified, but ECM not enabled on transmitter |
     | 636 | Binary File Transfer mode specified, but not supported by receiver |
     | 637 | Remote does not support EFF page options required by host |
     | 638 | Remote does not support EFF page coding |
     | 640 | No response to RR after three tries |
     | 641 | No response to CTC, or response was not CTR |
     | 642 | T5 time out since receiving first RNR |
     | 643 | Do not continue with next message after receiving ERR |
     | 644 | ERR response to EOR-EOP or EOR-PRI-EOP |
     | 645 | Transmitted DCN after receiving RTN |
     | 646 | EOR-MPS, EOR-EOM, EOR-NULL, EOR-PRI-MPS, or EOR-PRI-EOM sent after fourth PPR received |
     | 651 | RSPREC error |
     | 652 | No response to MPS, repeated three times |
     | 653 | Invalid response to MPS |
     | 654 | No response to EOP repeated three times |
     | 655 | Invalid response to EOP |
     | 656 | No response to EOM, repeated three times |
     | 657 | Invalid response to EOM |
     | 660 | DCN received in RSPREC |
     | 661 | No response received after third try for PPS-NULL |
     | 662 | No response received after third try for PPS-MPS |
     | 663 | No response received after third try for PPS-EOP |
     | 664 | No response received after third try for PPS-EOM |
     | 665 | No response received after third try for EOR-NULL |
     | 666 | No response received after third try for EOR-MPS |
     | 667 | No response received after third try for EOR-EOP |
     | 668 | No response received after third try for EOR-EOM |
     | 670 | Unspecified receive Phase B error |
     | 671 | RSPREC error |
     | 672 | COMREC error |
     | 673 | T.30 T2 timeout, expected page not received |
     | 674 | T.30 T1 timeout after EOM received |
     | 675 | DCN received in COMREC |
     | 676 | DCN received in RSPREC |
     | 677 | T.30 T2 timeout, expected page received |
     | 678 | Invalid DMA count specified for receiver |
     | 679 | Binary File Transfer specified, but ECM not supported by receiver |
     | 701 | RSPREC invalid response received |
     | 702 | COMREC invalid response received |
     | 703 | T3 timeout; no local response for remote voice interrupt |
     | 704 | T2 timeout; no command received after responding RNR |
     | 705 | DCN received for command received |
     | 706 | Command receive error |
     | 707 | Receive block count error in ECM mode |
     | 708 | Receive page count error in ECM mode |
     | 709 | EOR received in phase D |
     | 710 | Timeout while repeating RNR |
     | 750 | No EOL received in a five-second period |
     | 751 | Bad MMR data received from remote |
     | 840 | No interrupt acknowledge, timeout |
     | 841 | Loop current still present while playing recorder tone after timeout |
     | 842 | T.30 holdup timeout |
     | 843 | DCN received from host in receive holdup section for FAX PAD mode |
     | 844 | DCN received from host in receive holdup section for non-FAX PAD mode |
     | 858 | No dial tone detected |
     | 859 | No loop current detected |
     | 860 | Local phone in use successfully |
     | 861 | Busy trunk line detected |
     | 865 | T1 time slot busy |
     | 866 | Ringing detected during dialing |
     | 867 | Second or later wink missing for Feature Group D |
     | 901 | Normal busy; remote end busy (off-hook) |
     | 902 | Normal busy; remote end busy (off-hook in some countries) |
     | 903 | CNG fax calling tone detected |
     | 904 | Recall dial tone detected; signal generated when calling another party while already connected to one or more parties (for example, conference calling) |
     | 905 | Confirmation tone; automated equipment acknowledges successful completion of caller-requested feature (for example, call forwarding) |
     | 906 | This result is reserved |
     | 916 | Answer (probably human) detected; does not match any other expected call progress signal patterns |
     | 917 | Remote end answered call, but silence |
     | 918 | Dial tone detected; usually indicates the dialing sequence did not break dial tone |
     | 924 | No fax CNG tone detected after answering a call |
     | 925 | Indicates the remote end was ringing but did not answer |
     | 926 | Group 2 fax machine detected; remote machine is capable of sending and receiving G2 facsimiles only |
     | 927 | Intercept tone detected; remote end originating failure; invalid telephone number or class of service restriction |
     | 928 | After dialing the number, no energy detected on the line for the ced\_timeout timeout period; possible dead line |
     | 929 | Vacant tone detected; remote originating failure; invalid telephone number |
     | 930 | Reorder tone detected; end office (PBX) or carrier originating failure |
     | 931 | No circuit detected; end office or carrier originating failure, possible dead line |
     | 932 | CNG fax calling tone detected |
     | 939 | Fax machine detected; usually a fax CED tone, but also fax V.21 signals when the remote machine does not send a CED tone before it sends the fax |
     | 940 | Unknown problem (FCP\_UNKNOWN) |
     | 948 | ISDN channel call progress (FCP\_ISDN\_CALL\_PROGRESS) |
     | 949 | Indicates that a call collision occurred on the ISDN line |
     | 1100 | An error interrupt occurred, indicating a problem with the channel too severe to continue |
     | 1101 | The application was unable to process incoming interrupts/commands fast enough, and information was lost |
     | 1102 | The channel generated an unexpected 03 (reset done) or 7F interrupt, indicating the existence of a firmware or hardware problem |
     | 1103 | An API command to the driver returned an error value, indicating that the driver or the operating system detected an error |
     | 1104 | Error reported at termination of fax overlay download |
     | 1105 | Maximum timeout exceeded. This code occurs when the user configuration file parameter max timeout has been enabled and the specified timeout expires |
10. When you have finished entering information for the faxing setup, click **Finish**, or click **Cancel** to discard changes.  
       
     If you are configuring multiple Fax Server Clients, highlight the next **Fax Server Client** in the list and click the **Edit** button. Repeat steps 8-9.  
       
     To remove a fax setup, highlight the **Fax Server Client** and click the **Delete** button. This does not remove the Fax Server Client from the list, but it deletes all configuration information entered. To remove this item from the list, you will need to uninstall the Fax Server Client from the station running it.

  
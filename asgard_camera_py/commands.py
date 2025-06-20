# Command dictionary 

cred1_command_dict = {
    "all raw": "Display, colon-separated, camera parameters",
    "powers": "Get all camera powers",
    "powers raw": "raw printing",
    "powers getter": "Get getter power",
    "powers getter raw": "raw printing",
    "powers pulsetube": "Get pulsetube power",
    "powers pulsetube raw": "raw printing",
    "temperatures": "Get all camera temperatures",
    "temperatures raw": "raw printing",
    "temperatures motherboard": "Get mother board temperature",
    "temperatures motherboard raw": "raw printing",
    "temperatures frontend": "Get front end temperature",
    "temperatures frontend raw": "raw printing",
    "temperatures powerboard": "Get power board temperature",
    "temperatures powerboard raw": "raw printing",
    "temperatures water": "Get water temperature",
    "temperatures water raw": "raw printing",
    "temperatures ptmcu": "Get pulsetube MCU temperature",
    "temperatures ptmcu raw": "raw printing",
    "temperatures cryostat diode": "Get cryostat temperature from diode",
    "temperatures cryostat diode raw": "raw printing",
    "temperatures cryostat ptcontroller": "Get cryostat temperature from pulsetube controller",
    "temperatures cryostat ptcontroller raw": "raw printing",
    "temperatures cryostat setpoint": "Get cryostat temperature setpoint",
    "temperatures cryostat setpoint raw": "raw printing",
    "fps": "Get frame per second",
    "fps raw": "raw printing",
    "maxfps": "Get the max frame per second regarding current camera configuration",
    "maxfps raw": "raw printing",
    "peltiermaxcurrent": "Get peltiermaxcurrent",
    "peltiermaxcurrent raw": "raw printing",
    "ptready": "Get pulsetube ready information",
    "ptready raw": "raw printing",
    "pressure": "Get cryostat pressure",
    "pressure raw": "raw printing",
    "gain": "Get gain",
    "gain raw": "raw printing",
    "bias": "Get bias correction status",
    "bias raw": "raw printing",
    "flat": "Get flat correction status",
    "flat raw": "raw printing",
    "imagetags": "Get tags in image status",
    "imagetags raw": "raw printing",
    "led": "Get LED status",
    "led raw": "raw printing",
    "sendfile bias <bias image file size> <file MD5>": "Interpreter waits for bias image binary bytes; timeout restarts interpreter.",
    "sendfile flat <flat image file size> <file MD5>": "Interpreter waits for flat image binary bytes.",
    "getflat <url>": "Retrieve flat image from URL.",
    "getbias <url>": "Retrieve bias image from URL.",
    "gettestpattern <url>": "Retrieve test pattern images tar.gz file from URL for testpattern mode.",
    "testpattern": "Get testpattern mode status.",
    "testpattern raw": "raw printing",
    "events": "Camera events sending status",
    "events raw": "raw printing",
    "extsynchro": "Get external synchro usage status",
    "extsynchro raw": "raw printing",
    "rawimages": "Get raw images (no embedded computation) status",
    "rawimages raw": "raw printing",
    "getter nbregeneration": "Get getter regeneration count",
    "getter nbregeneration raw": "raw printing",
    "getter regremainingtime": "Get time remaining for getter regeneration",
    "getter regremainingtime raw": "raw printing",
    "cooling": "Get cooling status",
    "cooling raw": "raw printing",
    "standby": "Get standby mode status",
    "standby raw": "raw printing",
    "mode": "Get readout mode",
    "mode raw": "raw printing",
    "resetwidth": "Get reset width",
    "resetwidth raw": "raw printing",
    "nbreadworeset": "Get read count without reset",
    "nbreadworeset raw": "raw printing",
    "cropping": "Get cropping status (active/inactive)",
    "cropping raw": "raw printing",
    "cropping columns": "Get cropping columns config",
    "cropping columns raw": "raw printing",
    "cropping rows": "Get cropping rows config",
    "cropping rows raw": "raw printing",
    "aduoffset": "Get ADU offset",
    "aduoffset raw": "raw printing",
    "version": "Get all product versions",
    "version raw": "raw printing",
    "version firmware": "Get firmware version",
    "version firmware raw": "raw printing",
    "version firmware detailed": "Get detailed firmware version",
    "version firmware detailed raw": "raw printing",
    "version firmware build": "Get firmware build date",
    "version firmware build raw": "raw printing",
    "version fpga": "Get FPGA version",
    "version fpga raw": "raw printing",
    "version hardware": "Get hardware version",
    "version hardware raw": "raw printing",
    "status": (
        "Get camera status. Possible statuses:\n"
        "- starting: Just after power on\n"
        "- configuring: Reading configuration\n"
        "- poorvacuum: Vacuum between 10-3 and 10-4 during startup\n"
        "- faultyvacuum: Vacuum above 10-3\n"
        "- vacuumrege: Getter regeneration\n"
        "- ready: Ready to be cooled\n"
        "- isbeingcooled: Being cooled\n"
        "- standby: Cooled, sensor off\n"
        "- operational: Cooled, taking valid images\n"
        "- presave: Previous usage error occurred"
    ),
    "status raw": "raw printing",
    "status detailed": "Get last status change reason",
    "status detailed raw": "raw printing",
    "continue": "Resume camera if previously in error/poor vacuum state.",
    "save": "Save current settings; cooling/gain not saved.",
    "save raw": "raw printing",
    "ipaddress": "Display camera IP settings",
    "cameratype": "Display camera information",
    "exec upgradefirmware <url>": "Upgrade firmware from URL",
    "exec buildbias": "Build the bias image",
    "exec buildbias raw": "raw printing",
    "exec buildflat": "Build the flat image",
    "exec buildflat raw": "raw printing",
    "exec redovacuum": "Start vacuum regeneration",
    "set testpattern on": "Enable testpattern mode (loop of 32 images).",
    "set testpattern on raw": "raw printing",
    "set testpattern off": "Disable testpattern mode",
    "set testpattern off raw": "raw printing",
    "set fps <fpsValue>": "Set the frame rate",
    "set fps <fpsValue> raw": "raw printing",
    "set gain <gainValue>": "Set the gain",
    "set gain <gainValue> raw": "raw printing",
    "set bias on": "Enable bias correction",
    "set bias on raw": "raw printing",
    "set bias off": "Disable bias correction",
    "set bias off raw": "raw printing",
    "set flat on": "Enable flat correction",
    "set flat on raw": "raw printing",
    "set flat off": "Disable flat correction",
    "set flat off raw": "raw printing",
    "set imagetags on": "Enable tags in image",
    "set imagetags on raw": "raw printing",
    "set imagetags off": "Disable tags in image",
    "set imagetags off raw": "raw printing",
    "set led on": "Turn on LED; blinks purple if operational.",
    "set led on raw": "raw printing",
    "set led off": "Turn off LED",
    "set led off raw": "raw printing",
    "set events on": "Enable camera event sending (error messages)",
    "set events on raw": "raw printing",
    "set events off": "Disable camera event sending",
    "set events off raw": "raw printing",
    "set extsynchro on": "Enable external synchronization",
    "set extsynchro on raw": "raw printing",
    "set extsynchro off": "Disable external synchronization",
    "set extsynchro off raw": "raw printing",
    "set rawimages on": "Enable embedded computation on images",
    "set rawimages on raw": "raw printing",
    "set rawimages off": "Disable embedded computation",
    "set rawimages off raw": "raw printing",
    "set cooling on": "Enable cooling",
    "set cooling on raw": "raw printing",
    "set cooling off": "Disable cooling",
    "set cooling off raw": "raw printing",
    "set standby on": "Enable standby mode (cools camera, sensor off)",
    "set standby on raw": "raw printing",
    "set standby off": "Disable standby mode",
    "set standby off raw": "raw printing",
    "set mode globalreset": "Set global reset mode (legacy compatibility)",
    "set mode globalresetsingle": "Set global reset mode (single frame)",
    "set mode globalresetcds": "Set global reset correlated double sampling",
    "set mode globalresetbursts": "Set global reset multiple non-destructive readout mode",
    "set mode rollingresetsingle": "Set rolling reset (single frame)",
    "set mode rollingresetcds": "Set rolling reset correlated double sampling (compatibility)",
    "set mode rollingresetnro": "Set rolling reset multiple non-destructive readout",
    "set resetwidth <resetwidthValue>": "Set reset width",
    "set resetwidth <resetwidthValue> raw": "raw printing",
    "set nbreadworeset <nbreadworesetValue>": "Set read count without reset",
    "set nbreadworeset <nbreadworesetValue> raw": "raw printing",
    "set cropping on": "Enable cropping",
    "set cropping on raw": "raw printing",
    "set cropping off": "Disable cropping",
    "set cropping off raw": "raw printing",
    "set cropping columns <columnsValue>": "Set cropping columns selection; format: e.g., '1,3-9'.",
    "set cropping columns <columnsValue> raw": "raw printing",
    "set cropping rows <rowsValue>": "Set cropping rows selection; format: e.g., '1,3,9'.",
    "set cropping rows <rowsValue> raw": "raw printing",
    "set aduoffset <aduoffsetValue>": "Set ADU offset",
    "set aduoffset <aduoffsetValue> raw": "raw printing",
}


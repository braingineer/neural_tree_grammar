import logging
import os

def make_logger(igor, loggername):
    igor.logfile_level = 'debug'
    igor.logshell_level = 'info'
    igor.logfile_location = '.'
    logger = logging.getLogger(loggername)
    levels = {"debug": logging.DEBUG, "warning":logging.WARNING,
              "info": logging.INFO, "error":logging.ERROR,
              "critical":logging.CRITICAL}
    logger.setLevel(logging.DEBUG)

    if igor.logfile_level != "off":
        safe_loc = igor.logfile_location+("/" if igor.logfile_location[-1] != "/" else "")
        if not os.path.exists(safe_loc):
            os.makedirs(safe_loc)
        fh = logging.FileHandler(os.path.join(igor.logfile_location, 
                                              "{}.debug.log".format(loggername)))
        fh.setLevel(levels[igor.logfile_level])
        logger.addHandler(fh)

    if igor.logshell_level != "off":
        sh = logging.StreamHandler()
        sh.setLevel(levels[igor.logshell_level])
        logger.addHandler(sh)

    return logger
import os
import pandas as pd
import re

class LogLoader:
    '''
    # this part of code is from LogPai https://github.com/LogPai
    '''

    def __init__(self, log_format):
        self.logfile = None
        self.df_log = None
        self.log_format = log_format

    def format(self, logfile):
        self.logfile=logfile
        self.load_data()
        return self.df_log


    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', r'\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='UTF-8') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        # logdf.insert(0, 'LineId', None)
        # logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(self.logfile, regex, headers, self.log_format)

def wordsplit(log,dataset,regx=None,regx_use=False):

    if dataset == 'Android':
        log = re.sub(r'\(', '( ', log)
        log = re.sub(r'\)', ') ', log)
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    # elif dataset == 'Apache':
    #     log = re.sub(',', ', ', log)
    elif dataset == 'BGL':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
        log = re.sub(r'core\.', 'core. ', log)

    elif dataset == 'Hadoop':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        # log = re.sub(',', ', ', log)
    elif dataset == 'HDFS':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'HealthApp':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'HPC':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
        # log = re.sub('-', '- ', log)
        # log = re.sub(r'\[', '[ ', log)
        # log = re.sub(r'\]', '] ', log)

    elif dataset == 'Linux':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'Mac':
        log = re.sub(r'\[', '[ ', log)
        log = re.sub(r'\]', '] ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'OpenSSH':
        log = re.sub('=', '= ', log)
        log = re.sub(':', ': ', log)
        log = re.sub(',', ', ', log)
    # elif dataset == 'OpenStack':
    #     log = re.sub(',', ', ', log)
    elif dataset == 'Spark':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
        log = re.sub('/', '/ ', log)

    elif dataset == 'Proxifier':
        log = re.sub(r'\(.*?\)', '', log)
        log = re.sub(':', ' ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'Thunderbird':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
        log = re.sub('@', '@ ', log)
    elif dataset == 'Windows':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(r'\[', '[ ', log)
        log = re.sub(r'\]', '] ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'Zookeeper':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)

    if regx_use == True:
        for ree in regx:
            log = re.sub(ree, '<*>', log)

    logsplit = re.split(' +', log)
    return logsplit

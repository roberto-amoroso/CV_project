class Weight:
    def __init__(self, raw_name):
        self.name = raw_name
        self.params = {}
        params_list = raw_name[:-5].split('_')
        for p in params_list:
            couple = p.split('-')
            if couple[0] in ('name', 'net'):
                self.params[couple[0]] = couple[1]
            else:
                self.params[couple[0]] = int(couple[1])
        try:
            self.params['net']
            self.params['inputsize']
        except:
            raise ValueError

    def __getitem__(self, key):
        return self.params[key]
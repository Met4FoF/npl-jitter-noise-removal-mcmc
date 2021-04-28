class SineGeneratorJitter(DataStreamMET4FOF):
    """
    NPL addition to stream.py to add noise and jitter to sine generated data
    """
    def __init__(self, num_cycles=1000, jittersd=0.02, noisesd=0.05):
        #jittersd = 0.02
        xnn = np.arange(0, 3.142 * num_cycles, 0.1)
        xjitter = xnn + jittersd*np.random.randn(np.size(xnn))
        xn = np.sin(xjitter)
        xnlen = np.size(xn)
        x = xn + noisesd*np.random.randn(xnlen)
        self.set_data_source(x, y=None)



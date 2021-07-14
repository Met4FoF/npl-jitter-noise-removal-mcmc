from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
# from agentMET4FOF.streams import SineGeneratorJitter
from Sinegen import SineGeneratorJitter

import numpy as np
from NJRemove.NJRemoval_class_withmcmc import MCMCMH_NJ

def njr(fs, ydata, N, niter, tol, m0w, s0w, m0t, s0t, Mc, M0, Nc, Q):
    analyse_fun = MCMCMH_NJ(fs, ydata, N, niter, tol, m0w, s0w, m0t, s0t, Mc, M0, Nc, Q)
    yhat1= analyse_fun.AnalyseSignalN()
    return yhat1


########################################
class NJRemoved(AgentMET4FOF):
    def init_parameters(self, fs=100, ydata = np.array([]),  N=15, niter=100, tol=1e-9, m0w = 10, s0w = 0.0005, m0t = 10, s0t = 0.0002*100/8, Mc=5000, M0=100, Nc=100, Q=50 ):
        self.fs = fs
        self.ydata = ydata
        self.N = N
        self.niter = niter
        self.tol = tol
        self.m0w = m0w
        self.s0w = s0w
        self.m0t = m0t
        self.s0t = s0t
        self.Mc = Mc
        self.M0 = M0
        self.Nc = Nc
        self.Q = Q


    def on_received_message(self, message):
        ddata = message['data']
        self.ydata = np.append(self.ydata, ddata)
        if np.size(self.ydata) == self.N:
            t = njr(self.fs, self.ydata, self.N, self.niter, self.tol, self.m0w, self.s0w, self.m0t, self.s0t, self.Mc, self.M0,self.Nc, self.Q)
            self.send_output(self.ydata[7] - t)
            self.ydata = self.ydata[1:self.N]


class SineGeneratorAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = SineGeneratorJitter(jittersd=0.0005, noisesd=0.0002)

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample()  # dictionary
            self.send_output(sine_data['quantities'])


def main():
    # start agent network server
    agentNetwork = AgentNetwork(backend="mesa")
    # init agents

    gen_agent = agentNetwork.add_agent(agentType=SineGeneratorAgent)

    njremove_agent = agentNetwork.add_agent(agentType=NJRemoved)

    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    monitor_agent2 = agentNetwork.add_agent(agentType=MonitorAgent)


    # connect agents : We can connect multiple agents to any particular agent

    agentNetwork.bind_agents(gen_agent, njremove_agent)

    # connect
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # agentNetwork.bind_agents(njremove_agent, moitor_agent2)

    agentNetwork.bind_agents(njremove_agent, monitor_agent2)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
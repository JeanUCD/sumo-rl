import os
import sys


v =  '''<flow id="flow_ns_c" route="route_ns" begin="bb" end="fim" vehsPerHour="75" departSpeed="max" departLane="best"/>
    <flow id="flow_nw_c" route="route_nw" begin="bb" end="fim" vehsPerHour="75" departSpeed="max" departLane="best"/>
    <flow id="flow_ne_c" route="route_ne" begin="bb" end="fim" vehsPerHour="150" departSpeed="max" departLane="best"/>
    <flow id="flow_sw_c" route="route_sw" begin="bb" end="fim" vehsPerHour="150" departSpeed="max" departLane="best"/>
    <flow id="flow_sn_c" route="route_sn" begin="bb" end="fim" vehsPerHour="75" departSpeed="max" departLane="best"/>
    <flow id="flow_se_c" route="route_se" begin="bb" end="fim" vehsPerHour="75" departSpeed="max" departLane="best"/>

    <flow id="flow_en_c" route="route_en" begin="bb" end="fim" vehsPerHour="150" departSpeed="max" departLane="best"/>
    <flow id="flow_ew_c" route="route_ew" begin="bb" end="fim" vehsPerHour="150" departSpeed="max" departLane="best"/>
    <flow id="flow_es_c" route="route_es" begin="bb" end="fim" vehsPerHour="300" departSpeed="max" departLane="best"/>
    <flow id="flow_wn_c" route="route_wn" begin="bb" end="fim" vehsPerHour="300" departSpeed="max" departLane="best"/>
    <flow id="flow_we_c" route="route_we" begin="bb" end="fim" vehsPerHour="150" departSpeed="max" departLane="best"/>
    <flow id="flow_ws_c" route="route_ws" begin="bb" end="fim" vehsPerHour="150" departSpeed="max" departLane="best"/>'''

v =  '''<flow id="flow_ns_c" route="route_ns" begin="bb" end="fim" vehsPerHour="112.5" departSpeed="max" departLane="best"/>
    <flow id="flow_nw_c" route="route_nw" begin="bb" end="fim" vehsPerHour="112.5" departSpeed="max" departLane="best"/>
    <flow id="flow_ne_c" route="route_ne" begin="bb" end="fim" vehsPerHour="225" departSpeed="max" departLane="best"/>
    <flow id="flow_sw_c" route="route_sw" begin="bb" end="fim" vehsPerHour="225" departSpeed="max" departLane="best"/>
    <flow id="flow_sn_c" route="route_sn" begin="bb" end="fim" vehsPerHour="112.5" departSpeed="max" departLane="best"/>
    <flow id="flow_se_c" route="route_se" begin="bb" end="fim" vehsPerHour="112.5" departSpeed="max" departLane="best"/>

    <flow id="flow_en_c" route="route_en" begin="bb" end="fim" vehsPerHour="112.5" departSpeed="max" departLane="best"/>
    <flow id="flow_ew_c" route="route_ew" begin="bb" end="fim" vehsPerHour="112.5" departSpeed="max" departLane="best"/>
    <flow id="flow_es_c" route="route_es" begin="bb" end="fim" vehsPerHour="225" departSpeed="max" departLane="best"/>
    <flow id="flow_wn_c" route="route_wn" begin="bb" end="fim" vehsPerHour="225" departSpeed="max" departLane="best"/>
    <flow id="flow_we_c" route="route_we" begin="bb" end="fim" vehsPerHour="112.5" departSpeed="max" departLane="best"/>
    <flow id="flow_ws_c" route="route_ws" begin="bb" end="fim" vehsPerHour="112.5" departSpeed="max" departLane="best"/>'''


def get_context(begin, end, c):
    if c % 2 == 0:
        s = v
    else:
        s = h
    s = s.replace('c', str(c)).replace('bb', str(begin)).replace('fim', str(end))
    return s

def write_route_file(file, end, step):
    with open(file, 'w+') as f:
        f.write('''<routes>
                <route id="route_ns" edges="n_t t_s"/>
                <route id="route_nw" edges="n_t t_w"/>
                <route id="route_ne" edges="n_t t_e"/>
                <route id="route_we" edges="w_t t_e"/>
                <route id="route_wn" edges="w_t t_n"/>
                <route id="route_ws" edges="w_t t_s"/>
                <route id="route_ew" edges="e_t t_w"/>
                <route id="route_en" edges="e_t t_n"/>
                <route id="route_es" edges="e_t t_s"/>
                <route id="route_sn" edges="s_t t_n"/>
                <route id="route_se" edges="s_t t_e"/>
                <route id="route_sw" edges="s_t t_w"/>'''+'\n')

        c = 0
        for i in range(0, end, step):
            f.write(get_context(i, i+step, c))
            c += 1
        
        f.write('''</routes>''')

if __name__ == '__main__':
    write_route_file('nets/2way-single-intersection/single-intersection-gen-m-400k.rou.xml', 400000, 400000)

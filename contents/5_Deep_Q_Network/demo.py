# -*- codeing = utf-8 -*-
# @Time : 2020/12/14 16:34
# @Author : 王浩
# @File : demo.py
# @Software : PyCharm

from Sat_IoT_env import Sat_IoT

if __name__ == "__main__":
    sat_iot = Sat_IoT()
    sat_iot.__init__()
    sat_iot.show_system()
    sat_iot.step(222)
    sat_iot.show_system()
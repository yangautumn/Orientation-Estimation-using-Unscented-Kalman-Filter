#!/usr/bin/env python3.6

from ukf_note import *

for i in range(2, 4):
    print('\nRun on dataset {}'.format(i))
    run_ukf_on_dataset(i)

#!/usr/bin/env python3.6

from ukf_note import *

for i in range(1, 11):
    print('\nRun on dataset {}'.format(i))
    run_ukf_on_dataset(i)

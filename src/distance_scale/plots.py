import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import scipy.stats


def rurs_cropping_plot():
    # Total Laplacian runtime (no cropping vs automated cropping)
    patient_1_l_yc = np.mean([12.94 + 3.53, 14.04 + 3.56])
    patient_1_l_nc = (70.64 + 79.86)
    patient_2_l_yc = (11.14 + 2.68)
    patient_2_l_nc = (84.65 + 91.73)
    patient_3_l_yc = (11.1 + 2.01)
    patient_3_l_nc = (88.3 + 89.38)
    patient_4_l_yc = (15.42 + 3.84)
    patient_4_l_nc = (103.86 + 122.08)
    patient_5_l_yc = (22.35 + 9.55)
    patient_5_l_nc = (103.86 + 122.08)
    patient_6_l_yc = (21.32 + 8.02)
    patient_6_l_nc = (103.40 + 110.94)

    lap_yc = np.array([patient_1_l_yc, patient_2_l_yc, patient_3_l_yc, patient_4_l_yc, patient_5_l_yc, patient_6_l_yc])
    lap_nc = np.array([patient_1_l_nc, patient_2_l_nc, patient_3_l_nc, patient_4_l_nc, patient_5_l_nc, patient_6_l_nc])

    lap_yc_mean = np.mean(lap_yc)
    lap_nc_mean = np.mean(lap_nc)

    lap_yc_std = np.std(lap_yc)
    lap_nc_std = np.std(lap_nc)

    pval1 = scipy.stats.ttest_ind(lap_yc, lap_nc)[1]

    # Total time up through Butterworth (no cropping vs automated cropping)
    patient_1_b_yc = np.mean([89.48, 80.58])
    patient_1_b_nc = 489.28
    patient_2_b_yc = 92.12
    patient_2_b_nc = 623.77
    patient_3_b_yc = 94.92
    patient_3_b_nc = 414.53

    butter_yc = np.array([patient_1_b_yc, patient_2_b_yc, patient_3_b_yc])
    butter_nc = np.array([patient_1_b_nc, patient_2_b_nc, patient_3_b_nc])

    butter_yc_mean = np.mean(butter_yc)
    butter_nc_mean = np.mean(butter_nc)

    butter_yc_std = np.std(butter_yc)
    butter_nc_std = np.std(butter_nc)

    pval2 = scipy.stats.ttest_ind(butter_yc, butter_nc)[1]

    pairs = ['Laplacian Runtime \n(p = ' + str(np.round(pval1, 8)) + ')', 'Total Runtime up to Butterworth Filter \n(p = ' + str(np.round(
        pval2, 4)) + ')']
    x_pos = np.arange(len(pairs))
    CTEs_1 = [lap_yc_mean, butter_yc_mean]
    error_1 = [lap_yc_std, butter_yc_std]

    CTEs_2 = [lap_nc_mean, butter_nc_mean]
    error_2 = [lap_nc_std, butter_nc_std]

    fig, ax = plt.subplots()
    ax.bar(x_pos + 0.175, CTEs_1, yerr=error_1, alpha=0.5, width=0.35, ecolor='black', capsize=10, label='With Cropping')
    ax.bar(x_pos - 0.175, CTEs_2, yerr=error_2, alpha=0.5, width=0.35, ecolor='black', capsize=10, label='Without Cropping')
    #ax.legend(['With Cropping', 'Without Cropping'], fontsize=20)
    ax.set_ylabel('Runtime (sec)', fontsize=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pairs, fontsize=18)
    ax.set_title('Comparison of Runtime between Cropped and Non-Cropped Patient Videos (n=6)', fontsize=20)
    ax.legend()

    plt.savefig('RURS_plot.png')
    plt.show()


rurs_cropping_plot()
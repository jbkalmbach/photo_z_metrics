from scipy.stats import trim_mean


class calc_metrics(object):

    def __init__(self):

        return

    def calc_bins(self, z_est, z_true, z_max, n_bins):

        delta_z = (z_true - z_est) / (1. + z_true)
        bin_vals = [(float(i)/n_bins)*z_max for i in range(n_bins)]
        bin_vals.append(float(z_max))
        idx_sort = z_true.argsort()
        delta_z_sort = delta_z[idx_sort]
        z_true_sort = z_true[idx_sort]
        idx_bins = z_true_sort.searchsorted(bin_vals)
        delta_z_binned = [delta_z_sort[idx_bins[i]:idx_bins[i+1]] for i in range(n_bins)]

        return delta_z_binned

    def photo_z_bias(self, z_est, z_true, z_max, n_bins):

        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        bias_results = []
        for delta_z_data in delta_z_binned:
            trimmed_mean = trim_mean(delta_z_data, .25)
            bias_results.append(trimmed_mean)
        return np.array(bias_results)

    def photo_z_abs_bias(self, z_est, z_true, z_max, n_bins):

        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        bias_results = []
        for delta_z_data in delta_z_binned:
            trimmed_mean = trim_mean(np.abs(delta_z_data), .25)
            bias_results.append(trimmed_mean)
        return np.array(bias_results)

    def photo_z_stdev(self, z_est, z_true, z_max, n_bins):

        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        stdev_results = []
        for delta_z_data in delta_z_binned:
            bin_mean = np.mean(delta_z_data)
            diffs = delta_z_data - bin_mean
            diffs_sq_mean = np.mean(diffs**2.)
            stdev_results.append(np.sqrt(diffs_sq_mean))
        return np.array(stdev_results)

    def photo_z_stdev_iqr(self, z_est, z_true, z_max, n_bins):

        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        stdev_iqr_results = []
        for delta_z_data in delta_z_binned:
            bin_25 = np.percentile(delta_z_data, 25.)
            bin_75 = np.percentile(delta_z_data, 75.)
            diff = bin_75 - bin_25
            stdev_iqr_results.append(diff/1.349)
        return np.array(stdev_iqr_results)

    def photo_z_outlier_frac(self, z_est, z_true, z_max, n_bins):

        stdev_iqr_results = self.photo_z_stdev_iqr(z_est, z_true, z_max, n_bins)
        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        outlier_frac_results = []
        for delta_z_data, stdev_iqr_val in zip(delta_z_binned, stdev_iqr_results):
            if 3.*stdev_iqr_val < 0.06:
                outlier_thresh = 0.06
            else:
                outlier_thresh = 3.*stdev_iqr_val
            # print outlier_thresh, np.std(delta_z_data), len(delta_z_data), np.max(delta_z_data), np.min(delta_z_data)
            total_bin_obj = float(len(delta_z_data))
            outliers = np.where(np.abs(delta_z_data) > outlier_thresh)[0]
            # print outliers
            outlier_frac = len(outliers)/total_bin_obj
            outlier_frac_results.append(outlier_frac)
        return np.array(outlier_frac_results)

import numpy as np
from scipy.stats import trim_mean
from scipy.stats import kstest

__all__ = ["pointMetrics"]


class pointMetrics(object):

    def __init__(self):

        return

    def calc_bins(self, z_est, z_true, z_max, n_bins):

        """
        Sort the delta_z data into redshift bins in z_true. Delta_z is
        defined as (z_true - z_est) / (1. + z_true).

        Parameters
        ----------
        z_est: numpy array
          The photo-z point estimates.

        z_true: numpy array
          The true photo-z values.

        z_max: float
          The edge of the highest redshift bin wanted.

        n_bins: int
          The number of redshift bins between 0 and z_max.

        Output:
        -------
        delta_z_binned: numpy array
          The binned delta_z values.
        """

        delta_z = (z_true - z_est) / (1. + z_true)
        bin_vals = [(float(i)/n_bins)*z_max for i in range(n_bins)]
        bin_vals.append(float(z_max))
        idx_sort = z_true.argsort()
        delta_z_sort = delta_z[idx_sort]
        z_true_sort = z_true[idx_sort]
        idx_bins = z_true_sort.searchsorted(bin_vals)
        delta_z_binned = [delta_z_sort[idx_bins[i]:idx_bins[i+1]] for i in range(n_bins)]

        return np.array(delta_z_binned)

    # def calc_bins(self, z_est, z_true, z_max, n_bins):

    #     """
    #     Sort the delta_z data into redshift bins in z_true. Delta_z is
    #     defined as (z_true - z_est) / (1. + z_true).

    #     Parameters
    #     ----------
    #     z_est: numpy array
    #       The photo-z point estimates.

    #     z_true: numpy array
    #       The true photo-z values.

    #     z_max: float
    #       The edge of the highest redshift bin wanted.

    #     n_bins: int
    #       The number of redshift bins between 0 and z_max.

    #     Output:
    #     -------
    #     delta_z_binned: numpy array
    #       The binned delta_z values.
    #     """

    #     delta_z = (z_true - z_est) / (1. + z_true)
    #     bin_vals = list(np.percentile(z_true, np.linspace(0, 100, n_bins+1)))
    #     #bin_vals = [(float(i)/n_bins)*z_max for i in range(n_bins)]
    #     bin_vals.append(float(z_max))
    #     idx_sort = z_true.argsort()
    #     delta_z_sort = delta_z[idx_sort]
    #     z_true_sort = z_true[idx_sort]
    #     idx_bins = z_true_sort.searchsorted(bin_vals)
    #     delta_z_binned = [delta_z_sort[idx_bins[i]:idx_bins[i+1]] for i in range(n_bins)]

    #     return np.array(delta_z_binned)

    def photo_z_bias(self, z_est, z_true, z_max, n_bins):

        """
        Calculate the bias in each bin as a function of true redshift. Bias
        is the mean delta_z of the bin where delta_z is defined as
        (z_true - z_est) / (1. + z_true).

        Parameters
        ----------
        z_est: numpy array
          The photo-z point estimates.

        z_true: numpy array
          The true photo-z values.

        z_max: float
          The edge of the highest redshift bin wanted.

        n_bins: int
          The number of redshift bins between 0 and z_max.

        Output:
        -------
        bias_results: numpy array
          The bias as a function of true redshift.
        """

        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        bias_results = []
        for delta_z_data in delta_z_binned:
            bin_mean = np.mean(delta_z_data)
            bias_results.append(bin_mean)
        return np.array(bias_results)

    def photo_z_robust_bias(self, z_est, z_true, z_max, n_bins):

        """
        Calculate the robust bias in each bin as a function of true redshift.
        Robust bias is the trimmed mean delta_z of the bin where delta_z
        is defined as (z_true - z_est) / (1. + z_true) and we trim the highest
        and lowest 25% of values.

        Parameters
        ----------
        z_est: numpy array
          The photo-z point estimates.

        z_true: numpy array
          The true photo-z values.

        z_max: float
          The edge of the highest redshift bin wanted.

        n_bins: int
          The number of redshift bins between 0 and z_max.

        Output:
        -------
        bias_results: numpy array
          The robust bias as a function of true redshift.
        """

        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        bias_results = []
        for delta_z_data in delta_z_binned:
            trimmed_mean = trim_mean(delta_z_data, .25)
            bias_results.append(trimmed_mean)
        return np.array(bias_results)

    def photo_z_stdev(self, z_est, z_true, z_max, n_bins):

        """
        Calculate the standard deviation in each bin as a
        function of true redshift. Standard deviation is defined
        as the standard deviation of delta_z where delta_z
        is defined as (z_true - z_est) / (1. + z_true).

        Parameters
        ----------
        z_est: numpy array
          The photo-z point estimates.

        z_true: numpy array
          The true photo-z values.

        z_max: float
          The edge of the highest redshift bin wanted.

        n_bins: int
          The number of redshift bins between 0 and z_max.

        Output:
        -------
        stdev_results: numpy array
          The standard deviation as a function of true redshift.
        """

        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        stdev_results = []
        for delta_z_data in delta_z_binned:
            bin_mean = np.mean(delta_z_data)
            diffs = delta_z_data - bin_mean
            diffs_sq_mean = np.mean(diffs**2.)
            stdev_results.append(np.sqrt(diffs_sq_mean))
        return np.array(stdev_results)

    def photo_z_robust_stdev(self, z_est, z_true, z_max, n_bins):

        """
        Calculate the robust standard deviation in each bin as a
        function of true redshift. Robust standard deviation is defined
        as the standard deviation of delta_z in the bin where delta_z
        is defined as (z_true - z_est) / (1. + z_true) and we trim
        the highest and lowest 25% of delta_z values.

        Parameters
        ----------
        z_est: numpy array
          The photo-z point estimates.

        z_true: numpy array
          The true photo-z values.

        z_max: float
          The edge of the highest redshift bin wanted.

        n_bins: int
          The number of redshift bins between 0 and z_max.

        Output:
        -------
        stdev_results: numpy array
          The robust standard deviation as a function of true redshift.
        """

        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        stdev_iqr_results = []
        for delta_z_data in delta_z_binned:
            if len(delta_z_data) == 0:
                stdev_iqr_results.append(np.nan)
                continue
            bin_25 = np.percentile(delta_z_data, 25.)
            bin_75 = np.percentile(delta_z_data, 75.)
            diff = bin_75 - bin_25
            stdev_iqr_results.append(diff/1.349)
        return np.array(stdev_iqr_results)

    def photo_z_outlier_frac(self, z_est, z_true, z_max, n_bins):

        """
        Calculate the outlier fraction in each bin as a
        function of true redshift. Outlier fraction is defined
        as a delta_z value of more than 0.06 or 3 * (robust standard
        deviation) of the bin where definitions are the same as
        above.

        Parameters
        ----------
        z_est: numpy array
          The photo-z point estimates.

        z_true: numpy array
          The true photo-z values.

        z_max: float
          The edge of the highest redshift bin wanted.

        n_bins: int
          The number of redshift bins between 0 and z_max.

        Output:
        -------
        outlier_frac_results: numpy array
          The outlier fraction as a function of true redshift.
        """

        stdev_iqr_results = self.photo_z_robust_stdev(z_est, z_true, z_max, n_bins)
        delta_z_binned = self.calc_bins(z_est, z_true, z_max, n_bins)
        outlier_frac_results = []
        for delta_z_data, stdev_iqr_val in zip(delta_z_binned, stdev_iqr_results):
            if len(delta_z_data) == 0:
                outlier_frac_results.append(np.nan)
                continue
            #if 3.*stdev_iqr_val < 0.06:
            #    outlier_thresh = 0.06
            #if 3.*stdev_iqr_val < 0.1:
            #    outlier_thresh = 0.1
            #else:
            #    outlier_thresh = 3.*stdev_iqr_val
            outlier_thresh = 3.*0.05
            # print outlier_thresh, np.std(delta_z_data), len(delta_z_data), np.max(delta_z_data), np.min(delta_z_data)
            total_bin_obj = float(len(delta_z_data))
            outliers = np.where(np.abs(delta_z_data) > outlier_thresh)[0]
            # print outliers
            outlier_frac = len(outliers)/total_bin_obj
            outlier_frac_results.append(outlier_frac)
        return np.array(outlier_frac_results)


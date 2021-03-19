import numpy as np
from scipy.stats import trim_mean
from scipy.stats import kstest

__all__ = ["pdfMetrics"]


class pdfMetrics(object):

    def __init__(self):

        return

    def calc_pit(self, true_z, pdf_redshifts, pdf_z):

        """
        Parameters:
        -----------

        true_z: float
          The true redshift of the object.

        pdf_redshifts: numpy array
          The redshift values of the pdf array.

        pdf_z: numpy array
          The photo-z pdf at the redshifts of `pdf_redshifts`.

        Returns:
        --------

        pit_value: float
          The PIT value for the true_z, photo-z pdf input.
        """

        below_true_z = np.where(pdf_redshifts < true_z)

        pit_value = 0.
        pit_value += np.nansum(pdf_z[below_true_z])

        return pit_value

    def calc_pit_ks(self, pit_vals):

        """
        Return the value of a K-S test of the PIT distribution
        to a uniform distribution.

        Input
        -----
        pit_vals: numpy array
          The PIT values from `calc_pit`.

        Output
        ------
        scipy kstest output: scipy `KstestResult` object
          Returns an object with the test results and information.
          See scipy documentation for more.
        """

        return kstest(pit_vals, 'uniform')

    # def stack_pdfs(self, true_z, pdf_redshifts, pdf_z):

    #     stack_pdf = np.sum(pdf_z, axis=0)
    #     true_z_dist =

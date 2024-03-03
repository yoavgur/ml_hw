from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import intervals

class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """

        xs = np.random.rand(m)
        xs.sort()

        ys = []
        for x in xs:
            rand_num = np.random.rand()
            if (0 <= x <= 0.2) or (0.4 <= x <= 0.6) or (0.8 <= x <= 1):
                if rand_num <= 0.8:
                    ys.append(1)
                else:
                    ys.append(0)
            else:
                if rand_num <= 0.1:
                    ys.append(1)
                else:
                    ys.append(0)

        return np.column_stack((xs, ys))
                
    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        empirical_errors = np.array([0 for _ in range(m_first, m_last+1, step)], dtype=float)
        true_errors = np.array([0 for _ in range(m_first, m_last+1, step)], dtype=float)
        for _ in tqdm(range(T)):
            index = 0
            for m in range(m_first, m_last+1, step):
                data = self.sample_from_D(m)
                xs = data[:,0]
                ys = data[:,1]
                ints, empirical_error = intervals.find_best_interval(xs, ys, k)

                empirical_errors[index] += empirical_error / m
                true_errors[index] += self.calculate_true_error(ints)

                index += 1

        empirical_errors /= T
        true_errors /= T

        plt.plot(range(m_first, m_last+1, step), true_errors, label='True Error')
        plt.plot(range(m_first, m_last+1, step), empirical_errors, label='Empirical Error')
        
        plt.xlabel('m')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Empirical and True Error - m range')
        plt.show()


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        data = self.sample_from_D(m)
        xs = data[:,0]
        ys = data[:,1]

        empirical_errors = []
        true_errors = []
        for x in tqdm(range(k_first, k_last+1, step)):
            ints, empirical_error = intervals.find_best_interval(xs, ys, x)
            true_error = self.calculate_true_error(ints)

            empirical_errors.append(empirical_error / m)
            true_errors.append(true_error)

        plt.plot(range(k_first, k_last+1, step), true_errors, label='True Error')
        plt.plot(range(k_first, k_last+1, step), empirical_errors, label='Empirical Error')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Empirical and True Error - k range')
        plt.show()

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        data = self.sample_from_D(m)
        xs = data[:,0]
        ys = data[:,1]

        empirical_errors = []
        true_errors = []
        penalties = []
        sum_penalty_and_empirical_error = []
        for x in tqdm(range(k_first, k_last+1, step)):
            ints, empirical_error = intervals.find_best_interval(xs, ys, x)
            true_error = self.calculate_true_error(ints)

            empirical_errors.append(empirical_error / m)
            true_errors.append(true_error)
            penalties.append(self.penalty(x, m))
            sum_penalty_and_empirical_error.append(empirical_error / m + self.penalty(x, m))

        plt.plot(range(k_first, k_last+1, step), true_errors, label='True Error')
        plt.plot(range(k_first, k_last+1, step), empirical_errors, label='Empirical Error')
        plt.plot(range(k_first, k_last+1, step), penalties, label='Penalty')
        plt.plot(range(k_first, k_last+1, step), sum_penalty_and_empirical_error, label='Sum of Penalty and Empirical Error')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Empirical and True Error (SRM) - k range')
        plt.show()


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        data = self.sample_from_D(m)
        
        # Randomly split data 20% 80%
        np.random.shuffle(data)
        train, test = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]

        # Resort
        train = train[train[:,0].argsort()]
        test = test[test[:,0].argsort()]

        train_xs, train_ys = train[:,0], train[:,1]
        test_xs, test_ys = test[:,0], test[:,1]

        best_predictors = []
        for k in tqdm(range(1, 11)):
            ints, _ = intervals.find_best_interval(train_xs, train_ys, k)
            best_predictors.append(ints)

        best_error = 1
        best_predictor = best_predictors[0]
        for pred in tqdm(best_predictors):
            error = self.calculate_empirical_error_by_data(pred, np.column_stack((test_xs, test_ys)))
            if error < best_error:
                best_error = error
                best_predictor = pred

        
        return len(best_predictor) # best k value

    def penalty(self, k, n):
        delta = 0.1
        return 2 * (((2*k + np.log(2/delta)) / (n)) ** 0.5)

    def calculate_true_error(self, l_intervals):
        true_ints = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        true_inverts = [(0.2, 0.4), (0.6, 0.8)]

        def get_interval_overlap(start1, end1, start2, end2):
            return max(0, min(end1, end2) - max(start1, start2))

        def get_overlap(start, end, ints):
            true_overlap = 0
            for start2, end2 in ints:
                true_overlap += get_interval_overlap(start, end, start2, end2)

            return true_overlap

        def get_inverted_intervals(intervals):
            inverted_intervals = [(0, intervals[0][0])]
            
            for x in range(len(intervals)-1):
                inverted_intervals.append((intervals[x][1], intervals[x+1][0]))

            inverted_intervals.append((intervals[-1][1], 1))

            return inverted_intervals

        inverted_intervals = get_inverted_intervals(l_intervals)
        overlapped_positive = sum([get_overlap(start, end, true_ints) for start, end in l_intervals])
        overlapped_negative = sum([get_overlap(start, end, true_inverts) for start, end in inverted_intervals])

        not_overlapped_positive = 0.6 - overlapped_positive
        not_overlapped_negative = 0.4 - overlapped_negative

        return (
            0.2 * overlapped_positive + # Expected error even if they matched the true intervals
            0.8 * not_overlapped_positive + # Expected error if they didn't match the true intervals
            0.1 * overlapped_negative + # Expected error even if they didn't match anything they shouldn't have
            0.9 * not_overlapped_negative # Expected error if the did match something they shouldn't have
        )

    def calculate_empirical_error_by_data(self, l_intervals, data):
        errors = 0
        for x,y in data:
            found = False

            for start, end in l_intervals:
                if start <= x <= end:
                    found = True
                    if y == 0:
                        errors += 1
                    break

            if not found and y == 1:
                errors += 1
                


        return errors / len(data)



if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)


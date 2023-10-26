import socket
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.signal import find_peaks

import scipy.stats as stats
from sklearn.cluster import DBSCAN
import os
import dpkt

def parse_pcap(file_name):
    timestamps = []
    packet_sizes = []
    with open(file_name, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        count = 0
        for timestamp, packet in pcap:
            if count == 0:
                start_time = timestamp
            eth = dpkt.ethernet.Ethernet(packet)
            if isinstance(eth.data, dpkt.ip.IP):  # check if this is an IP packet
                ip = eth.data
                packet_size = ip.len  # get the size of the packet
                timestamps.append(timestamp-start_time)
                packet_sizes.append(packet_size)
            count += 1
    return timestamps, packet_sizes

def extract_conv_features(timestamps, packet_sizes, filename, action_name, kernel):
    signal = np.array(packet_sizes)

    signal = signal / np.linalg.norm(signal)
    kernel = kernel / np.linalg.norm(kernel)

    result = np.convolve(signal, kernel, mode ='same')

    # Plot the convolved result versus the timestamps
    plt.figure(figsize=(10,6))
    plt.plot(timestamps[:len(result)], result)
    plt.xlabel('Timestamp')
    plt.ylabel('Convolved Packet Sizes')
    plt.title('Convolution Result vs Timestamp')
    
    directory_path = os.path.join("plots", action_name, "conv")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    output_file_path = os.path.join(directory_path, filename + '.png')
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

    # Extracting statistical features from the convolution result
    conv_mean = np.mean(result)
    conv_std = np.std(result)
    conv_median = np.median(result)
    conv_percentile_25 = np.percentile(result, 25)
    conv_percentile_75 = np.percentile(result, 75)
    conv_max = np.max(result)
    conv_min = np.min(result)
    conv_skewness = stats.skew(result)
    conv_kurtosis = stats.kurtosis(result)

    # Thresholding and Clustering
    threshold = 0.25
    match_indices = np.where(result > threshold)[0]
    match_timestamps = [timestamps[i] for i in match_indices]

    if len(match_timestamps) != 0:

        # Total time from start of first cluster to end of last cluster
        total_time = match_timestamps[-1] - match_timestamps[0]    
        # Compute the time gaps between consecutive clusters
        time_gaps = [match_timestamps[i+1] - match_timestamps[i] for i in range(len(match_timestamps)-1)]
        # Average time gap between clusters
        average_time_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
    else:
        total_time = 0
        average_time_gap = 0
    # Gather all the computed statistics into a dictionary
    statistics = {
        'conv_mean': conv_mean,
        'conv_std': conv_std,
        'conv_median': conv_median,
        'conv_percentile_25': conv_percentile_25,
        'conv_percentile_75': conv_percentile_75,
        'conv_max': conv_max,
        'conv_min': conv_min,
        'conv_skewness': conv_skewness,
        'conv_kurtosis': conv_kurtosis,
        'total_clusters': len(match_timestamps),
        'total_time_span': total_time,
        'average_time_gap': average_time_gap
    }

    return statistics

def extract_corr_coeff_features(timestamps, packet_sizes, filename, action_name, kernel):
    signal = np.array(packet_sizes)

    signal = signal / np.linalg.norm(signal)
    kernel = kernel / np.linalg.norm(kernel)

    result = np.convolve(signal, kernel, mode ='same')

    window_size = len(kernel)
    correlation_coefficients = []

    for i in range(len(signal) - window_size + 1):
        window = signal[i:i+window_size]
        correlation_coefficients.append(np.corrcoef(window, kernel)[0, 1])

    # plot the convolved result versus the timestamps
    plt.figure(figsize=(10,6))
    plt.plot(timestamps[:len(correlation_coefficients)], correlation_coefficients)
    plt.xlabel('Timestamp')
    plt.ylabel('Correlation Coefficient')
    plt.title('Correlation Coefficient Result vs Timestamp')
    directory_path = os.path.join("plots", action_name, "corr_coeff")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    output_file_path = os.path.join(directory_path, filename + '.png')
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

    corr_mean = np.mean(correlation_coefficients)
    corr_std = np.std(correlation_coefficients)
    corr_median = np.median(correlation_coefficients)
    corr_percentile_25 = np.percentile(correlation_coefficients, 25)
    corr_percentile_75 = np.percentile(correlation_coefficients, 75)
    corr_max = np.max(correlation_coefficients)
    corr_min = np.min(correlation_coefficients)
    corr_skewness = stats.skew(correlation_coefficients)
    corr_kurtosis = stats.kurtosis(correlation_coefficients)
    # Compute mean and standard deviation of correlation coefficients
    mean_corr = np.mean(correlation_coefficients)
    std_corr = np.std(correlation_coefficients)

    # Define a match as a point where the correlation coefficient is more than 2 standard deviations from the mean
    threshold = mean_corr + 1.2 * std_corr

    # Find the timestamps where the correlation coefficients exceed the threshold
    match_indices = np.where(correlation_coefficients > threshold)[0]
    match_timestamps = [timestamps[i] for i in match_indices]
    
    clusters = []
    current_cluster = [match_timestamps[0]]

    for i in range(1, len(match_timestamps)):
        if match_timestamps[i] - current_cluster[-1] <= 0.05:
            current_cluster.append(match_timestamps[i])
        else:
            if len(current_cluster) >= 10:
                clusters.append(current_cluster)
            current_cluster = [match_timestamps[i]]

    # Check if the last cluster also satisfies the condition
    if len(current_cluster) >= 10:
        clusters.append(current_cluster)

    # 'clusters' is a list of lists, where each sub-list is a group of adjacent points
    # Flatten it into a single list for DBSCAN
    filtered_match_timestamps = [ts for cluster in clusters for ts in cluster]
    if len(filtered_match_timestamps) != 0:

        # Apply DBSCAN
        db = DBSCAN(eps=0.5, min_samples=11).fit(np.array(filtered_match_timestamps).reshape(-1, 1))

        # Get labels
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Now we can extract where each cluster is around and their length information

        clusters = {}
        for cluster_id in set(labels):
            if cluster_id == -1:  # it's noise
                continue
            samples_in_cluster = np.array(filtered_match_timestamps)[labels == cluster_id]
            clusters[cluster_id] = {
                "start": np.min(samples_in_cluster),
                "end": np.max(samples_in_cluster),
                "length": np.max(samples_in_cluster) - np.min(samples_in_cluster),
            }

        # Total number of clusters
        num_clusters = len(clusters)

        # Lengths of all clusters
        lengths = [info['length'] for info in clusters.values()]

        # Sum of all cluster lengths
        total_length = sum(lengths)

        if num_clusters != 0:
            # Average length of clusters
            average_length = total_length / num_clusters
            sorted_clusters = sorted(clusters.values(), key=lambda x: x['start'])

            # Total time from start of first cluster to end of last cluster
            total_time = sorted_clusters[-1]['end'] - sorted_clusters[0]['start']

            # Time gaps between consecutive clusters
            time_gaps = [sorted_clusters[i+1]['start'] - sorted_clusters[i]['end'] for i in range(len(sorted_clusters)-1)]

            # Average time gap between clusters
            average_time_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
        else:
            num_clusters = 0
            total_length = 0
            average_length = 0
            total_time = 0
            average_time_gap = 0
    else:
        num_clusters = 0
        total_length = 0
        average_length = 0
        total_time = 0
        average_time_gap = 0
    # Gather all the computed statistics into a dictionary
    statistics = {
        'corr_mean': corr_mean,
        'corr_std': corr_std,
        'corr_median': corr_median,
        'corr_percentile_25': corr_percentile_25,
        'corr_percentile_75': corr_percentile_75,
        'corr_max': corr_max,
        'corr_min': corr_min,
        'corr_skewness': corr_skewness,
        'corr_kurtosis': corr_kurtosis,
        'num_clusters': num_clusters,
        'total_length': total_length,
        'average_length': average_length,
        'total_time': total_time,
        'average_time_gap': average_time_gap,
    }

    return statistics

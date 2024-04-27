import sys
import chdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze(bench_results):
    df = chdb.query(f"""
    SELECT 
        toUInt64(extract(id, '/.*/(\d+)')) as cardinality,
        extract(id, '/(.*)/') as estimator,
        round(mean.estimate / if(cardinality = 0, 1, cardinality), 2) as time
    FROM
        '{bench_results}'
    WHERE 
        match(id, 'insert')
    ORDER BY
        cardinality
    """, 'DataFrame');
    render_comparison(df, 'insert', 'time', 'linear')

    df = chdb.query(f"""
    SELECT 
        toUInt64(extract(id, '/.*/(\d+)')) as cardinality,
        extract(id, '/(.*)/') as estimator,
        round(mean.estimate, 2) as time
    FROM
        '{bench_results}'
    WHERE 
        match(id, 'estimate')
    ORDER BY
        cardinality
    """, 'DataFrame');

    render_comparison(df, 'estimate', 'time', 'log')

    df = pd.read_table('target/memory_allocations.md', delimiter='|', comment='-', skipinitialspace=True)
    df = df.dropna(axis=1, how='all').rename(columns=lambda x: x.strip().replace('_', '-')).dropna()
    for column in df.columns[1:]:
        df[column] = df[column].apply(lambda x: sum(int(num) for num in x.split('/')[:2]))
    df_memory = df.melt(id_vars='cardinality', var_name='estimator', value_name='bytes')
    render_comparison(df_memory, 'memory', 'bytes', 'log')


def render_comparison(df, operation, metric, yscale):
    pivot_df = df.pivot(index='cardinality', columns='estimator', values=metric)

    desired_order = ['cardinality-estimator', 'amadeus-streaming']
    rest_of_columns = [col for col in pivot_df.columns if col not in desired_order]
    final_column_order = desired_order + rest_of_columns
    pivot_df = pivot_df[final_column_order]

    md_path = f'target/{operation}_{metric}.md'
    pivot_df.apply(highlight_min, axis=1).to_markdown(md_path, tablefmt='github')
    print(f"saved markdown table to {md_path}")

    plt.figure(figsize=(12, 8))

    colors = {
        'cardinality-estimator': 'green',
        'amadeus-streaming': 'blue',
        'probabilistic-collections': 'red',
        'hyperloglog': 'purple',
        'hyperloglogplus': 'orange',
        # ... define other colors for other estimators if there are any ...
    }

    palette_dict = {est: colors[est] for est in df['estimator'].unique() if est in colors}

    sns.scatterplot(data=df, x='cardinality', y=metric, hue='estimator', style='estimator', palette=palette_dict, s=100)

    for i, estimator in enumerate(df['estimator'].unique()):
        subset = df[df['estimator'] == estimator]
        plt.plot(subset['cardinality'], subset[metric], linestyle='--', linewidth=0.5, color=palette_dict[estimator])

    plt.xscale('log', base=2)
    plt.yscale(yscale)

    x_ticks_estimate = df['cardinality'].unique()
    selected_x_ticks_estimate = [x for x in x_ticks_estimate if x > 0]
    x_labels_estimate = [str(int(x)) if x > 0 else '0' for x in selected_x_ticks_estimate]

    plt.xticks(selected_x_ticks_estimate, labels=x_labels_estimate, rotation=45)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.xlabel('Cardinality')
    plt.ylabel(metric)

    plot_path = f'target/{operation}_{metric}.png'
    plt.savefig(plot_path)
    print(f"saved plot to {plot_path}")


# Function to highlight the minimum value in each row (size)
def highlight_min(row):
    min_val = row.min()
    return row.apply(lambda x: '**' + str(x) + '**' if x == min_val else str(x))


if __name__ == '__main__':
    analyze(sys.argv[1])

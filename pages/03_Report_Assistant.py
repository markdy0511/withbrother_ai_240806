from datetime import datetime, timedelta
import calendar
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import insert_logo
import pandas as pd
from langchain.schema import StrOutputParser
from PIL import Image

st.set_page_config(
    page_title="Report Assistant",
    page_icon="🐥",
    layout="wide",
)


insert_logo.add_logo("withbrother_logo.png")

overview_llm = ChatOpenAI(
    temperature=0.7,
    model = "gpt-4o"
)

media_llm = ChatOpenAI(
    temperature=0.1,
    model = "gpt-4o"
)

strict_llm = ChatOpenAI(
    temperature=0.3,
    model = "gpt-4o"
)

influence_llm = ChatOpenAI(
    temperature=0.5,
    model = "gpt-4o"
)

# GPT-4 모델 초기화
image_llm = ChatOpenAI(model="gpt-4o")

sort_orders = {
    '노출수': False,  # 내림차순
    '클릭수': False,  # 내림차순
    'CTR': False,  # 내림차순'
    'CPC': True,  # 오름차순
    '총비용': False,  # 내림차순
    '회원가입': False,  # 내림차순
    'DB전환': False,  # 내림차순
    '가망': False,  # 내림차순
    '전환수': False,  # 내림차순
    'CPA': True,  # 오름차순
}

metric = ['노출수','클릭수', 'CTR','CPC', '총비용', '회원가입',
    'DB전환','가망', '전환수', 'CPA']

review = []

# 파일 입력기
def load_data(file):
    if file is not None:
        try:
            try:
                data = pd.read_csv(file)
            except:
                data = pd.read_excel(file)
        except:
            data = pd.read_csv(history_file, encoding='cp949')
        return data
    return None

# 주차 계산기
def get_week_info(date, start_weekday):

    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
        date = date.date()
    # Define the start of the week (0 = Monday, 6 = Sunday)
    weekday_dict = {"월요일": 0, "일요일": 6}
    start_weekday_num = weekday_dict[start_weekday]
    # Calculate the start of the week for the given date
    start_of_week = date - timedelta(days=(date.weekday() - start_weekday_num) % 7)

    # Get the month and the week number
    month = start_of_week.month
    start_of_month = datetime(start_of_week.year, month, 1).date()
    week_number = ((start_of_week - start_of_month).days // 7) + 1
    
    # Get the month name in Korean for output
    month_dict_kr = {
        1: "1월", 2: "2월", 3: "3월", 4: "4월", 5: "5월", 6: "6월", 
        7: "7월", 8: "8월", 9: "9월", 10: "10월", 11: "11월", 12: "12월"
    }
    month_name_kr = month_dict_kr[month]
    
    return str(month_name_kr)+" "+str(week_number)+"주"

# 월 계산기
def get_month_info(date):
    return date.month

# 기간 tagger
def filter_by_period(df, period_type, reference_date, start_of_week):
    #reference_date = pd.to_datetime(reference_date)
    
    if period_type == '일간':
        filtered_df = df[(df['일자'] >= reference_date - timedelta(days=1)) & (df['일자'] <= reference_date)]
        now = reference_date
        pre = reference_date - timedelta(days=1)
        return filtered_df, now, pre
    elif period_type == '주간':
        this_week = get_week_info(reference_date, start_of_week)
        pre_week = get_week_info(reference_date - timedelta(days=7), start_of_week)
        df['주'] = df['일자'].apply(lambda x: get_week_info(x, start_of_week))
        
        filtered_weeks = [pre_week, this_week]
        filtered_df = df[df['주'].isin(filtered_weeks)]

        now = this_week
        pre = pre_week
        return filtered_df, now, pre

    elif period_type == '월간':
        this_month = get_month_info(reference_date)
        pre_month = this_month - 1
        df['주'] = df['일자'].apply(lambda x: get_week_info(x, start_of_week))
        df['월'] = df['일자'].apply(lambda x: get_month_info(x))

        filtered_months = [pre_month, this_month]
        filtered_df = df[df['월'].isin(filtered_months)]

        now = this_month
        pre = pre_month
        return filtered_df, now, pre

    else:
        raise ValueError("Invalid period_type. It should be one of ['일간', '주간', '월간']")


# 지표 추출
def process_dataframes(df1, df2, commerce_or_not, analysis_period):
    if commerce_or_not == "비커머스":
        if analysis_period == "일간":
            exclude_column = ["일자", "캠페인", "광고그룹", "소재명/키워드", "디바이스", "매체", "소재구분", "소재종류", "광고유형"]
        elif analysis_period == "주간":
            exclude_column = ["일자", "캠페인", "광고그룹", "소재명/키워드", "디바이스", "매체", "소재구분", "소재종류", "광고유형","주"]
        else:
            exclude_column = ["일자", "캠페인", "광고그룹", "소재명/키워드", "디바이스", "매체", "소재구분", "소재종류", "광고유형","주","월"]
    else: #커머스
        if analysis_period == "일간":
            exclude_column = ["일자", "캠페인", "광고그룹", "소재명/키워드", "디바이스", "매체", "광고유형"]
        elif analysis_period == "주간":
            exclude_column = ["일자", "캠페인", "광고그룹", "소재명/키워드", "디바이스", "매체", "광고유형","주"]
        else:
            exclude_column = ["일자", "캠페인", "광고그룹", "소재명/키워드", "디바이스", "매체", "광고유형","주","월"]

    # 특정 열을 제외한 나머지 열을 리스트로 변환
    list_media = df1.drop(columns=exclude_column).columns.tolist()
    list_ga = df2.drop(columns=exclude_column).columns.tolist()
    
    # 매체 데이터의 유입, 전환 지표 분리
    list_inflow = [item for item in list_media if item in ["노출수","클릭수","총비용"]]
    list_trans_media = [item for item in list_media if item not in ["노출수","클릭수","총비용"]]
    
    return list_inflow, list_trans_media, list_ga

# 매체 report 데이터 생성기
def report_table(df, list_inflow, list_trans_media, selected_trans, commerce_or_not):
    if commerce_or_not == "비커머스":
        columns_inflow = list_inflow + ['CTR','CPC']
        columns_trans = list_trans_media + ['전환수','CPA']
        columns_report = columns_inflow + columns_trans
    else: #커머스
        columns_inflow = list_inflow + ['CTR','CPC']
        columns_trans = list_trans_media + ['전환수','객단가','CPA','ROAS','전환율']
        columns_report = columns_inflow + columns_trans
    
    report_df = pd.DataFrame(columns=columns_report)
    report_df = pd.concat([report_df, df])
    
    # ZeroDivisionError 예외 처리
    report_df['CTR'] = report_df.apply(lambda row: (row['클릭수'] / row['노출수'] * 100) if row['노출수'] != 0 else 0, axis=1)
    report_df['CPC'] = report_df.apply(lambda row: (row['총비용'] / row['클릭수']) if row['클릭수'] != 0 else 'INF', axis=1)
    
    report_df['전환수'] = report_df[selected_trans].sum(axis=1) #report_df['회원가입'] + report_df['DB전환'] + report_df['가망']
    report_df['CPA'] = report_df.apply(lambda row: (row['총비용'] / row['전환수']) if row['전환수'] != 0 else 'INF', axis=1)
    
    # 데이터 타입 확인 및 변환
    report_df['CTR'] = pd.to_numeric(report_df['CTR'], errors='coerce')
    report_df['CPC'] = pd.to_numeric(report_df['CPC'], errors='coerce')
    report_df['CPA'] = pd.to_numeric(report_df['CPA'], errors='coerce')
    
    report_df['CTR'] = report_df['CTR'].round(2)
    report_df['CPC'] = report_df['CPC'].round(0)
    report_df['CPA'] = report_df['CPA'].round(0)

    if commerce_or_not == "커머스":
        report_df['객단가'] = report_df.apply(lambda row: (row['구매액'] / row['구매']) if row['구매'] != 0 else 0, axis=1)
        report_df['ROAS'] = report_df.apply(lambda row: (row['구매액'] / row['총비용'] * 100) if row['총비용'] != 0 else 0, axis=1)
        report_df['전환율'] = report_df.apply(lambda row: (row['전환수'] / row['클릭수'] * 100) if row['클릭수'] != 0 else 0, axis=1)

        # 데이터 타입 확인 및 변환
        report_df['객단가'] = pd.to_numeric(report_df['객단가'], errors='coerce')
        report_df['ROAS'] = pd.to_numeric(report_df['ROAS'], errors='coerce')
        report_df['전환율'] = pd.to_numeric(report_df['전환율'], errors='coerce')
        
        report_df['객단가'] = report_df['객단가'].round(0)
        report_df['ROAS'] = report_df['ROAS'].round(0)
        report_df['전환율'] = report_df['전환율'].round(2)
    
    return report_df

# ga report 데이터 생성기
def ga_report_table(df, list_trans_ga, selected_trans, commerce_or_not):
    if commerce_or_not == "비커머스":
        columns_trans = list_trans_ga + ['전환수','CPA']
    else: #커머스
        columns_trans = list_trans_ga + ['전환수','객단가','CPA','ROAS','전환율']
    
    report_df = pd.DataFrame(columns=columns_trans)
    report_df = pd.concat([report_df, df])

    report_df['전환수'] = report_df[selected_trans].sum(axis=1) #report_df['회원가입'] + report_df['db전환'] + report_df['카톡btn'] + report_df['전화btn']

    if commerce_or_not == "커머스":
        report_df['객단가'] = report_df.apply(lambda row: (row['구매액'] / row['구매']) if row['구매'] != 0 else 0, axis=1)

        # 데이터 타입 확인 및 변환
        report_df['객단가'] = pd.to_numeric(report_df['객단가'], errors='coerce')
        
        report_df['객단가'] = report_df['객단가'].round(0)

    return report_df

def calculate_percentage_change(df1, df2, column):
    changes = []
    for i in range(len(df1)):
        if df1[column].iloc[i] != 0:
            change = round(((df2[column].iloc[i] - df1[column].iloc[i]) / df1[column].iloc[i]) * 100, 2)
            changes.append(change)
        else:
            changes.append(None)  # or 0, if you prefer to mark 0 change
    return changes
    #return round(((df2[column] - df1[column]) / df1[column]) * 100, 2)

# Function to calculate new influence
def calculate_new_influence(df1, df2, column):
    total_change = abs(df2[column].sum() - df1[column].sum())
    return (df2[column] - df1[column]) / total_change

# 증감율 계산 함수
def calculate_increase_rate(row_3, row_2):
    return round(((row_3 - row_2) / row_2) * 100, 2) if row_2 != 0 else 'None'

def generate_statements(df, now_ch_cmp_week, metrics, top_num):
    statements = []
        # Statements for sum metrics

    metrics = [element for element in metrics if (element != '총비용') and (element != '전환수')]
    for metric in metrics:
        if metric in ['CPA', 'CPC', 'CTR']:
            if metric == 'CPA':
                top_10_cost = df['총비용'].sum()
                top_10_acquisitions = df['전환수'].sum()
                total_cost = now_ch_cmp_week['총비용'].sum()
                total_acquisitions = now_ch_cmp_week['전환수'].sum()
                top_10_metric = top_10_cost / top_10_acquisitions if top_10_acquisitions != 0 else 0
                total_metric = total_cost / total_acquisitions if total_acquisitions != 0 else 0
            elif metric == 'CPC':
                top_10_cost = df['총비용'].sum()
                top_10_clicks = df['클릭수'].sum()
                total_cost = now_ch_cmp_week['총비용'].sum()
                total_clicks = now_ch_cmp_week['클릭수'].sum()
                top_10_metric = top_10_cost / top_10_clicks if top_10_clicks != 0 else 0
                total_metric = total_cost / total_clicks if total_clicks != 0 else 0
            elif metric == 'CTR':
                top_10_clicks = df['클릭수'].sum()
                top_10_impressions = df['노출수'].sum()
                total_clicks = now_ch_cmp_week['클릭수'].sum()
                total_impressions = now_ch_cmp_week['노출수'].sum()
                top_10_metric = (top_10_clicks / top_10_impressions) * 100 if top_10_impressions != 0 else 0
                total_metric = (total_clicks / total_impressions) * 100 if total_impressions != 0 else 0

            ratio = round((top_10_metric - total_metric),2)
            statement = f"정렬된 상위 {top_num}개의 {metric} ({top_10_metric:.2f})는 당 기간 전체 {metric} ({total_metric:.2f})보다 {ratio}만큼 차이가 있습니다."
            statements.append(statement)
        else:
            top_10_sum = df[metric].sum()
            total_sum = now_ch_cmp_week[metric].sum()
            ratio = round((top_10_sum / total_sum) * 100, 2)
            statement = f"정렬된 상위 {top_num}개의 {metric} ({top_10_sum:,})는 당 기간 전체 {metric} ({total_sum:,})의 {ratio}% 입니다."
            statements.append(statement)

    return statements




#보고서 유형 저장
if 'condition_set' not in st.session_state:
    st.session_state.condition_set = None

#원천 데이터 저장
if 'media_df' not in st.session_state:
    st.session_state.media_df = None

if 'original_ga_df' not in st.session_state:
    st.session_state.original_ga_df = None

if 'original_history_df' not in st.session_state:
    st.session_state.original_history_df = None

#가공 데이터 저장
if 'internal_ch_df' not in st.session_state:
    st.session_state.internal_ch_df = None

if 'now_media' not in st.session_state:
    st.session_state.now_media = None

if 'pre_media' not in st.session_state:
    st.session_state.pre_media = None

if 'ga_df' not in st.session_state:
    st.session_state.ga_df = None

if 'now_ga' not in st.session_state:
    st.session_state.now_ga = None

if 'pre_ga' not in st.session_state:
    st.session_state.pre_ga = None

if 'history_df' not in st.session_state:
    st.session_state.history_df = None

if 'now_history' not in st.session_state:
    st.session_state.now_history = None

if 'pre_history' not in st.session_state:
    st.session_state.pre_history = None

#원천 지표 저장
if 'list_inflow' not in st.session_state:
    st.session_state.list_inflow = None

if 'list_trans_media' not in st.session_state:
    st.session_state.list_trans_media = None

if 'list_trans_ga' not in st.session_state:
    st.session_state.list_trans_ga = None

#가공 지표 저장
if 'trans_metric_set' not in st.session_state:
    st.session_state.trans_metric_set = None

#오버뷰 보고서 결과 저장
if 'overview_result' not in st.session_state:
    st.session_state.overview_result = None

if 'overview_chain_result' not in st.session_state:
    st.session_state.overview_chain_result = None

#기간 합산 결과 저장
if 'overview_ad_df_result' not in st.session_state:
    st.session_state.overview_ad_df_result = None

if 'overview_ga_ad_df_result' not in st.session_state:
    st.session_state.overview_ga_ad_df_result = None

#매체별 분석 보고서 결과 저장
if 'ch_ranking_result' not in st.session_state:
    st.session_state.ch_ranking_result = None

if 'ch_ranking_chain_result' not in st.session_state:
    st.session_state.ch_ranking_chain_result = None

if 'ch_ranking_influence_analysis' not in st.session_state:
    st.session_state.ch_ranking_influence_analysis = None

if 'ch_ranking_individual_results' not in st.session_state:
    st.session_state.ch_ranking_individual_results = {}

#소재별 분석 보고서 결과 저장
if 'br_ranking_result' not in st.session_state:
    st.session_state.br_ranking_result = None

if 'br_ranking_chain_result' not in st.session_state:
    st.session_state.br_ranking_chain_result = None

if 'br_ranking_influence_analysis' not in st.session_state:
    st.session_state.br_ranking_influence_analysis = None

if 'br_ranking_individual_results' not in st.session_state:
    st.session_state.br_ranking_individual_results = {}

#소재구분별 소재종류 분석 보고서 결과 저장
if 'selected_br' not in st.session_state:
    st.session_state.selected_br = None

if 'selected_metric_br' not in st.session_state:
    st.session_state.selected_metric_br = []

if 'sorted_df_br' not in st.session_state:
    st.session_state.sorted_df_br = None

if 'br_statements' not in st.session_state:
    st.session_state.br_statements = []

if 'br_detail_chain_result' not in st.session_state:
    st.session_state.br_detail_chain_result = None

# 캠페인 분석
if 'selected_ad_type' not in st.session_state:
    st.session_state.selected_ad_type = None

if 'selected_media_cmp' not in st.session_state:
    st.session_state.selected_media_cmp = None

if 'selected_metric_cmp' not in st.session_state:
    st.session_state.selected_metric_cmp = []

if 'sorted_df_cmp' not in st.session_state:
    st.session_state.sorted_df_cmp = None

if 'cmp_statements' not in st.session_state:
    st.session_state.cmp_statements = []

if 'cmp_ranking_chain_result' not in st.session_state:
    st.session_state.cmp_ranking_chain_result = None

# 그룹 분석
if 'selected_campaign_cmp' not in st.session_state:
    st.session_state.selected_campaign_cmp = None

if 'sorted_df_grp' not in st.session_state:
    st.session_state.sorted_df_grp = None

if 'grp_statements' not in st.session_state:
    st.session_state.grp_statements = []

if 'grp_ranking_chain_result' not in st.session_state:
    st.session_state.grp_ranking_chain_result = None

# Streamlit app layout
st.title('보고서 작성 도우미')

# 데이터 입력기
with st.sidebar: #원하는 소스를 만드는 곳
    st.sidebar.header('이곳에 데이터를 업로드하세요.')
    
    media_file = st.file_uploader(
        "매체 데이터 업로드 (Excel or CSV)",
        type=['xls','xlsx', 'csv'],
        key="uploader1"
    )
    ga_file = st.file_uploader(
        "GA 데이터 업로드 (Excel or CSV)",
        type=['xls','xlsx', 'csv'],
        key="uploader2"
    )

    history_file = st.file_uploader(
        "운영 히스토리 데이터 업로드 (Excel or CSV)",
        type=["xls", "xlsx", "csv"],
        key="uploader3"
    )


# 보고서 유형 선택
if st.session_state.condition_set is None:
    with st.form(key='condition_form'):
        # Include ROAS analysis
        commerce_or_not = st.radio(
            "광고주가 커머스 분야인가요? 아니면 비커머스 분야인가요? (필수)",
            ("커머스", "비커머스")
        )

        # Select analysis period
        analysis_period = st.radio(
            "분석할 기간을 선택하세요 (필수)",
            ("일간", "주간", "월간")
        )
        selected_date = st.date_input(
            "분석 시작 날짜를 선택해주세요. 주간, 월간일 경우 포함 날짜 아무 일이나 선택해주세요. (필수)",
            datetime.today(), key="selected_date"
        )

        week_start_day = st.radio(
            "주의 시작 요일을 선택하세요. 주간 분석을 하지 않을 경우 아무것이나 선택해도 됩니다. (선택)",
                ("월요일", "일요일")
            )

        # 조건 버튼 입력
        submit_condition = st.form_submit_button(label='설정 완료')

        if submit_condition:
            st.session_state.condition_set = {'commerce_or_not': commerce_or_not, 'analysis_period': analysis_period, 'selected_date':selected_date, 'week_start_day':week_start_day}

#이미 보고서 유형을 선택했을 경우
else:
    with st.form(key='condition_form'):
        # Include ROAS analysis
        option_1 = ["커머스", "비커머스"]
        initial_selection_1 = st.session_state.condition_set["commerce_or_not"]
        initial_index_1 = option_1.index(initial_selection_1)

        commerce_or_not = st.radio(
            "광고주가 커머스 분야인가요? 아니면 비커머스 분야인가요? (필수)",
            ("커머스", "비커머스"), index=initial_index_1
        )

        option_2 = ["일간", "주간", "월간"]
        initial_selection_2 = st.session_state.condition_set["analysis_period"]
        initial_index_2 = option_2.index(initial_selection_2)
        # Select analysis period
        analysis_period = st.radio(
            "분석할 기간을 선택하세요 (필수)",
            ("일간", "주간", "월간"), index=initial_index_2
        )

        initial_date = st.session_state.condition_set["selected_date"]
        selected_date = st.date_input(
            "분석 시작 날짜를 선택해주세요. 주간, 월간일 경우 포함 날짜 아무 일이나 선택해주세요. (필수)",
            key="selected_date", value=initial_date
        )


        option_4 = ["월요일", "일요일"]
        initial_selection_4 = st.session_state.condition_set["week_start_day"]
        initial_index_4 = option_4.index(initial_selection_4)
        week_start_day = st.radio(
            "주의 시작 요일을 선택하세요. 주간 분석을 하지 않을 경우 아무것이나 선택해도 됩니다. (선택)",
                ("월요일", "일요일"), index=initial_index_4
            )

        # 조건 버튼 입력
        submit_condition = st.form_submit_button(label='설정 완료')

        if submit_condition:
            st.session_state.condition_set = {'commerce_or_not': commerce_or_not, 'analysis_period': analysis_period, 'selected_date':selected_date, 'week_start_day':week_start_day}

# 최초 보고서 유형 제출 및 파일 업로드 완료
if st.session_state.condition_set and (st.session_state.media_df is None) and (st.session_state.original_ga_df is None) and (st.session_state.history_df is None):
    commerce_or_not = st.session_state.condition_set['commerce_or_not']
    analysis_period = st.session_state.condition_set['analysis_period']
    selected_date = st.session_state.condition_set['selected_date']
    week_start_day = st.session_state.condition_set['week_start_day']

    if analysis_period == "일간":
        st.write(selected_date, " 을(를) 기준으로 전 일과 비교 분석 합니다.")
    elif analysis_period == "주간":
        st.write(get_week_info(selected_date,week_start_day), " 을(를) 기준으로 전 주와 비교 분석 합니다.")
    else:
        st.write(get_month_info(selected_date), " 을(를) 기준으로 전 월과 비교 분석 합니다.")
    
    with st.spinner("데이터 가져오는 중..."):
        media_df = load_data(media_file)
        original_ga_df = load_data(ga_file)
        original_ga_df['일자'] = pd.to_datetime(original_ga_df['일자'], format='%Y-%m-%d')
        original_history_df = load_data(history_file)
        st.session_state.media_df = media_df
        st.session_state.original_ga_df = original_ga_df
        st.session_state.original_history_df = original_history_df
        
        internal_ch_df,now_media, pre_media = filter_by_period(media_df, analysis_period, selected_date, week_start_day)
        st.session_state.internal_ch_df = internal_ch_df
        st.session_state.now_media = now_media
        st.session_state.pre_media = pre_media

        ga_df,now_ga, pre_ga = filter_by_period(original_ga_df, analysis_period, selected_date, week_start_day)
        st.session_state.ga_df = ga_df
        st.session_state.now_ga = now_ga
        st.session_state.pre_ga = pre_ga

        history_df,now_history, pre_history = filter_by_period(original_history_df, analysis_period, selected_date, week_start_day)
        st.session_state.history_df = history_df
        st.session_state.now_history = now_history
        st.session_state.pre_history = pre_history
        
        list_inflow, list_trans_media, list_trans_ga = process_dataframes(internal_ch_df, ga_df, commerce_or_not, analysis_period)
        st.session_state.list_inflow = list_inflow
        st.session_state.list_trans_media = list_trans_media
        st.session_state.list_trans_ga = list_trans_ga

# 이미 업로드한 경우
elif st.session_state.condition_set and (st.session_state.media_df is not None) and (st.session_state.original_ga_df is not None) and (st.session_state.history_df is not None):
    commerce_or_not = st.session_state.condition_set['commerce_or_not']
    analysis_period = st.session_state.condition_set['analysis_period']
    selected_date = st.session_state.condition_set['selected_date']
    week_start_day = st.session_state.condition_set['week_start_day']

    if analysis_period == "일간":
        st.write(selected_date, " 을(를) 기준으로 전 일과 비교 분석 합니다.")
    elif analysis_period == "주간":
        st.write(get_week_info(selected_date,week_start_day), " 을(를) 기준으로 전 주와 비교 분석 합니다.")
    else:
        st.write(get_month_info(selected_date), " 을(를) 기준으로 전 월과 비교 분석 합니다.")
    
    with st.spinner("데이터 가져오는 중..."):
        media_df = st.session_state.media_df
        original_ga_df = st.session_state.original_ga_df
        original_history_df = st.session_state.original_history_df

        internal_ch_df = st.session_state.internal_ch_df
        now_media = st.session_state.now_media
        pre_media = st.session_state.pre_media

        ga_df = st.session_state.ga_df
        now_ga = st.session_state.now_ga
        pre_ga = st.session_state.pre_ga

        history_df = st.session_state.history_df
        now_history = st.session_state.now_history
        pre_history = st.session_state.pre_history

        list_inflow = st.session_state.list_inflow
        list_trans_media = st.session_state.list_trans_media
        list_trans_ga = st.session_state.list_trans_ga

# 보고서 유형이나 파일이 제출되지 않은 상태
else:
    st.write("1. 사이드 바에 매체, GA, 운영 데이터 파일을 업로드하고, 보고서 유형을 선택해 설정 완료 버튼을 눌러주세요.")


# 전환 지표 설정 전
if st.session_state.condition_set and (st.session_state.trans_metric_set is None):
    st.title("성과 보고서")
    with st.form(key='metric_select_form'):
        selected_trans_media = st.multiselect("매체 데이터에서 전환의 총합으로 사용될 지표들을 선택해주세요.", list_trans_media)
        selected_trans_ga = st.multiselect("GA 데이터에서 전환의 총합으로 사용될 지표들을 선택해주세요.", list_trans_ga)
        # 조건 버튼 입력
        submit_trans = st.form_submit_button(label='설정 완료')
        if submit_trans:
            st.session_state.trans_metric_set = {'selected_trans_media': selected_trans_media, 'selected_trans_ga': selected_trans_ga}
# 전환 지표 설정 후
elif st.session_state.condition_set and st.session_state.trans_metric_set:
    st.title("성과 보고서")
    with st.form(key='metric_select_form'):
        default_values_1 = st.session_state.trans_metric_set["selected_trans_media"]
        selected_trans_media = st.multiselect("매체 데이터에서 전환의 총합으로 사용될 지표들을 선택해주세요.", list_trans_media, default=default_values_1)
        default_values_2 = st.session_state.trans_metric_set["selected_trans_ga"]
        selected_trans_ga = st.multiselect("GA 데이터에서 전환의 총합으로 사용될 지표들을 선택해주세요.", list_trans_ga, default=default_values_2)
        # 조건 버튼 입력
        submit_trans = st.form_submit_button(label='설정 완료')
        if submit_trans:
            st.session_state.trans_metric_set = {'selected_trans_media': selected_trans_media, 'selected_trans_ga': selected_trans_ga}
# 보고서 유형 설정 전
else:   
    st.write("2. 파일 업로드와 설정 완료 버튼을 누르면, 전환 지표 설정 창이 생깁니다.")


# 보고서 생성 시작
if st.session_state.trans_metric_set:
    selected_trans_media = st.session_state.trans_metric_set['selected_trans_media']
    selected_trans_ga = st.session_state.trans_metric_set['selected_trans_ga']
    target_list_media = list_inflow + list_trans_media
    with st.spinner("보고서 초안 생성 중..."):
        #기간 그룹핑용
        if analysis_period == "일간":
            group_period = "일자"
        elif analysis_period == "주간":
            group_period = "주"
        else:
            group_period = "월"
    #비커머스
    if commerce_or_not == "비커머스":
        overview, ch_ranking, brnch_ranking, brnch_detail_ranking, cmp_ranking, grp_ranking, kwrd_ranking, history, preview = st.tabs(["오버뷰","매체별 성과","소재구분 분석","소재종류 분석","매체 선택 캠페인 분석", "캠페인 선택 그룹 분석", "성과 상위 소재(키워드) 분석", '운영 히스토리',  '프리뷰'])
        internal_ch_df['일자'] = internal_ch_df['일자'].astype(str)

        with overview:
            if st.session_state.overview_result is None:
                st.subheader('오버뷰')
                with st.spinner('데이터 분석 중...'):
                    target_list_media = list_inflow + list_trans_media

                    result = {}
                    for index, row in internal_ch_df.iterrows():
                        category = row[group_period]
                        
                        if category not in result:
                            result[category] = {col: 0 for col in target_list_media}
                        
                        for col in target_list_media:
                            result[category][col] += row[col]
                    ad_week = pd.DataFrame(result).T
                    ad_week.index.name = group_period

                    cal_ad_week = report_table(ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)
                    st.session_state.overview_ad_df_result = cal_ad_week
                    cal_ad_week.loc['변화량'] = cal_ad_week.diff().iloc[1]
                    cal_ad_week.loc['증감율'] = round(((cal_ad_week.loc[now_media] - cal_ad_week.loc[pre_media]) / cal_ad_week.loc[pre_media]) * 100, 2)

                    result_ga = {}
                    for index, row in ga_df.iterrows():
                        category = row[group_period]
                        
                        if category not in result_ga:
                            result_ga[category] = {col: 0 for col in list_trans_ga}
                        
                        for col in list_trans_ga:
                            result_ga[category][col] += row[col]
                    ga_ad_week = pd.DataFrame(result_ga).T
                    ga_ad_week.index.name = group_period

                    ga_cal_ad_week = ga_report_table(ga_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)

                    ga_cal_ad_week['CPA'] = (cal_ad_week['총비용'] / ga_cal_ad_week['전환수'])
                    ga_cal_ad_week['CPA'] = pd.to_numeric(ga_cal_ad_week['CPA'], errors='coerce')
                    ga_cal_ad_week['CPA'] = ga_cal_ad_week['CPA'].round(0)

                    ga_cal_ad_week.columns = [f'GA_{col}' for col in ga_cal_ad_week.columns]

                    st.session_state.overview_ga_ad_df_result = ga_cal_ad_week

                    ga_cal_ad_week.loc['변화량'] = ga_cal_ad_week.diff().iloc[1]
                    ga_cal_ad_week.loc['증감율'] = round(((ga_cal_ad_week.loc[now_media] - ga_cal_ad_week.loc[pre_media]) / ga_cal_ad_week.loc[pre_media]) * 100, 2)

                    

                    # 데이터 프레임을 좌우로 붙이기
                    df_combined = pd.concat([cal_ad_week, ga_cal_ad_week], axis=1)
                    st.session_state.overview_result = df_combined
                    st.write(df_combined)

                description = "Periodical change data results:\n\n"
                description += df_combined.to_string()

                previous_period = df_combined.iloc[0]
                current_period = df_combined.iloc[1]
                change_period = df_combined.iloc[2]
                columns = df_combined.columns[1:]

                # Generating the sentences
                sentences = []
                for col in columns:
                    change = "증가" if change_period[col] > 0 else "감소"
                    sentence = f"{col}은 지난 기간 대비 {abs(change_period[col]):,.2f} {change}하였습니다. ({previous_period[col]:,.2f} -> {current_period[col]:,.2f})"
                    sentences.append(sentence)

                
            
                month_compare_prompt = ChatPromptTemplate.from_template(
                    """
                    너는 퍼포먼스 마케팅 성과 분석가야.
                    다음 주차에 따른 성과 자료를 기반으로 유입 성과와 전환 성과를 분석해야해.
                    \n\n{description}
                    \n\n{sentences}

                    노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                    회원가입, DB전환, 가망, 전환수, CPA는 매체 전환에 대한 성과야.
                    GA_회원가입, GA_db전환, GA_카톡btn, GA_전화btn, GA_총합계, GA_CPA는 GA 전환에 대한 성과야.

                    첫 행은 비교하기 위한 바로 직전 기간의 성과이고, 두번째 행은 이번 기간의 성과야.

                    유입 성과는 CTR과 CPC가 얼마나 변하였고, 그에 대한 근거로 노출수와 클릭수, 비용이 어떻게 변화했기에 CTR과 CPC가 그러한 변화를 가지게 되었는지 분석해야해.
                    전환 성과는 전환수가 얼마나 변하였고, CPA가 얼마나 변하였는지를 파악하고, 그에 대한 근거로 노출수, 클릭수, 비용, 회원가입, DB전환, 가망에서의 변화를 분석해야해.
                    매체 전환과 GA 전환을 구분해서 설명해야해.

                    숫자를 사용할 때는 지난 기간의 절대값과 이번 기간의 절대값을 모두 표시해줘.
                    증감율에서 숫자를 인용할 때는 퍼센테이지를 붙여서 설명해야해.
                    1% 이상의 변화가 있을 때는 유지된 것이 아닌, 어떤 이유로 증가되었는지 또는 감소되었는지를 분석해야해.
                    비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대해.
                    비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대하는 것 잊지마.
                    증감율이 양수면 증가, 음수면 감소야.

                    아래 예시를 잘 참고해서 작성해줘.
                    1번 예시
                    - 지난주에 대비하여 전환수는 유지하였으나, 전체적으로 광고 성과가 감소한 추세입니다.
                    - 유입 성과에 관련하여, 전주 대비 지출된 비용의 증가로 노출수는 증가하였지만, 클릭수가 감소하면서 CTR은 2% 감소(100 -> 98)하였으며, CPC는 5% 증가 (1,000 -> 1,050)하였습니다.
                    - 매체 전환 성과에 관련하여, 전주 대비 전환수는 유지되었으나, 지출 비용의 증가로 CPA가 증가하였습니다.(100 -> 140)
                    - GA 전환 성과에 관련하여, 전주 대비 전환수는 유지되었으나, 지출 비용의 증가로 CPA가 증가하였습니다.(100 -> 138)
                    - 전반적으로 감소된 유입에 비해, 전환이 유지되면서 구체화된 타겟층을 발견한 점은 고무적이며, 클릭수와 전환수를 증가시키는데 노력하고자 합니다.

                    2번 예시
                    - 지난주에 대비하여 전환수가 증가하였지만, 유입 성과가 감소하였습니다.
                    - 유입 성과에 관련하여, 전주 대비 지출된 비용을 증가하였지만, 노출수와 클릭수가 감소하며 CTR은 감소폭에 비해, CPC가 20%로 크게 증가(1,000 -> 1,200)하였습니다.
                    - 매체 전환 성과에 관련하여, 전주 대비 회원가입의 증가로 전환수는 소폭 증가하였지만, 지출 비용의 증가폭이 더 크기 때문에 CPA가 5% (100 -> 105)증가하였습니다.
                    - GA 전환 성과에 관련하여, 전주 대비 회원가입의 증가로 전환수는 소폭 증가하였지만, 지출 비용의 증가폭이 더 크기 때문에 CPA가 5% (100 -> 105)증가하였습니다.
                    - 전반적으로 유입 성과가 감소한 상황에서 전환 성과가 증가한 것은 긍정적이며, 클릭수의 증가와 전환수의 증가폭를 늘리는 방향의 전략이 필요합니다.

                    분석 결과를 5줄로 출력해줘.
                    완벽한 인과관계를 설명하면 너에게 보상을 줄게.

                    데이터에서 잘못읽으면 패널티가 있어.
                    

                """
                )

                comparison_month_chain = month_compare_prompt | overview_llm | StrOutputParser()
                with st.status("전체 요약 분석...") as status: 
                    descript = comparison_month_chain.invoke(
                        {"description": description,"sentences":sentences},
                    )
                    st.session_state.overview_chain_result = descript

                review.append(descript)
                sentences = descript.split('.\n')
                bullet_list = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences if sentence) + "</ul>"
                st.markdown(bullet_list, unsafe_allow_html=True)
            else:
                st.subheader('오버뷰')
                st.write(st.session_state.overview_result)
                sentences = st.session_state.overview_chain_result.split('.\n')
                bullet_list = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences if sentence) + "</ul>"
                st.markdown(bullet_list, unsafe_allow_html=True)

        with ch_ranking:
            if st.session_state.ch_ranking_result is None:
                with st.spinner('매체별 데이터...'):
                    result = {}
                    for index, row in internal_ch_df.iterrows():
                        key = (row['매체'], row[group_period])
                        
                        if key not in result:
                            result[key] = {col: 0 for col in target_list_media}
                        
                        for col in target_list_media:
                            result[key][col] += row[col]

                    # 결과를 데이터프레임으로 변환
                    ch_ad_week = pd.DataFrame(result).T
                    ch_ad_week.index.names = ['매체', group_period]
                    
                    cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                    result_ga = {}
                    for index, row in ga_df.iterrows():
                        key = (row['매체'], row[group_period])
                        
                        if key not in result_ga:
                            result_ga[key] = {col: 0 for col in list_trans_ga}
                        
                        for col in list_trans_ga:
                            result_ga[key][col] += row[col]

                    # 결과를 데이터프레임으로 변환
                    ga_ch_ad_week = pd.DataFrame(result_ga).T
                    ga_ch_ad_week.index.names = ['매체', group_period]
                    
                    ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                    
                    ga_cal_ch_ad_week['CPA'] = (cal_ch_ad_week['총비용'] / ga_cal_ch_ad_week['전환수'])
                    ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                    ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)
                    
                    ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                    df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                    df_combined.reset_index(inplace=True)
                    df_combined[['매체', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                    df_combined.drop(columns=['index'], inplace=True)
                    # 특정 열을 앞에 오도록 열 순서 재배치
                    columns = ['매체', group_period] + [col for col in df_combined.columns if (col != '매체') and (col != group_period)]
                    df_combined_re = df_combined[columns]

                    result = {}
                    cal_ad_week = st.session_state.overview_ad_df_result
                    ga_cal_ad_week = st.session_state.overview_ga_ad_df_result

                    sum_df_combined = pd.concat([cal_ad_week, ga_cal_ad_week], axis=1)
                    

                    st.subheader('기간별 매체 순위 변화')
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader(pre_media)
                        pre_week = df_combined_re[df_combined_re[group_period] == pre_media]
                        pre_week_desc = pre_week.sort_values(by='전환수', ascending=False)
                        # Step 2: Extract a row using loc
                        row = sum_df_combined.loc[pre_media]  # This extracts the second row (index 1)

                        # Convert the extracted row to a DataFrame
                        row_df = pd.DataFrame([row])

                        # Optionally reset the index to maintain consistency
                        row_df.reset_index(drop=True, inplace=True)

                        # Step 3: Concatenate the DataFrame with the extracted row
                        pre_result = pd.concat([pre_week_desc, row_df], axis=0, ignore_index=True)
                        pre_description = "Previous period performance data results:\n\n"
                        pre_description += pre_week_desc.to_string()
                        st.write(pre_result)
                        result['pre_period'] = pre_result
                    with col2:
                        st.subheader(now_media)
                        now_week = df_combined_re[df_combined_re[group_period] == now_media]
                        now_week_desc = now_week.sort_values(by='전환수', ascending=False)
                        # Step 2: Extract a row using loc
                        row = sum_df_combined.loc[now_media]  # This extracts the second row (index 1)

                        # Convert the extracted row to a DataFrame
                        row_df = pd.DataFrame([row])

                        # Optionally reset the index to maintain consistency
                        row_df.reset_index(drop=True, inplace=True)

                        # Step 3: Concatenate the DataFrame with the extracted row
                        now_result = pd.concat([now_week_desc, row_df], axis=0, ignore_index=True)
                        now_description = "This period performance data results:\n\n"
                        now_description += now_week_desc.to_string()
                        st.write(now_result)
                        result['now_period'] = now_result
                    st.session_state.ch_ranking_result = result
                    percentage_changes = {
                        "매체": pre_result["매체"],
                        "노출수 변화율": calculate_percentage_change(pre_result, now_result, "노출수"),
                        "클릭수 변화율": calculate_percentage_change(pre_result, now_result, "클릭수"),
                        "CTR 변화율": calculate_percentage_change(pre_result, now_result, "CTR"),
                        "CPC 변화율": calculate_percentage_change(pre_result, now_result, "CPC"),
                        "총비용 변화율": calculate_percentage_change(pre_result, now_result, "총비용"),
                        "회원가입 변화율": calculate_percentage_change(pre_result, now_result, "회원가입"),
                        "DB전환 변화율": calculate_percentage_change(pre_result, now_result, "DB전환"),
                        "가망 변화율": calculate_percentage_change(pre_result, now_result, "가망"),
                        "전환수 변화율": calculate_percentage_change(pre_result, now_result, "전환수"),
                        "CPA 변화율": calculate_percentage_change(pre_result, now_result, "CPA")
                    }

                    df_percentage_changes = pd.DataFrame(percentage_changes)
                    df_per_description = "Periodical change data results by channel:\n\n"
                    df_per_description += df_percentage_changes.to_string()

                    # Calculate new influences
                    influences = {
                        "매체": pre_result["매체"],
                        "노출수 영향도": calculate_new_influence(pre_result, now_result, "노출수"),
                        "클릭수 영향도": calculate_new_influence(pre_result, now_result, "클릭수"),
                        "총비용 영향도": calculate_new_influence(pre_result, now_result, "총비용"),
                        "전환수 영향도": calculate_new_influence(pre_result, now_result, "전환수")
                    }

                    df_influences = pd.DataFrame(influences)

                    # Calculate new impact changes
                    impact_changes = {
                        "매체": df_percentage_changes["매체"],
                        "노출수 영향 변화율": df_influences["노출수 영향도"] * df_percentage_changes["노출수 변화율"],
                        "클릭수 영향 변화율": df_influences["클릭수 영향도"] * df_percentage_changes["클릭수 변화율"],
                        "총비용 영향 변화율": df_influences["총비용 영향도"] * df_percentage_changes["총비용 변화율"],
                        "전환수 영향 변화율": df_influences["전환수 영향도"] * df_percentage_changes["전환수 변화율"]
                    }

                    df_impact_changes = pd.DataFrame(impact_changes)

                    df_impact_description = "Periodical change data results influencing by channel:\n\n"
                    df_impact_description += df_impact_changes.to_string()

                    #매체별 성과 증감 비교
                    dic_ch_ad_week = {}
                    dic_description = {}
                    channels = now_week_desc['매체'].unique()

                    for channel in channels:
                        ch_df = df_combined_re[df_combined_re['매체'] == str(channel)]
                        ch_df.set_index(group_period, inplace=True)
                        ch_df.drop(columns=['매체'], inplace=True)

                        ch_df.loc['변화량'] = ch_df.diff().iloc[1]
                        # 새로운 증감율 행 생성
                        increase_rate = []
                        for col in ch_df.columns:
                            rate = calculate_increase_rate(ch_df.loc[now_media, col], ch_df.loc[pre_media, col])
                            increase_rate.append(rate)

                        # 데이터프레임에 증감율 행 추가
                        ch_df.loc['증감율'] = increase_rate
                        #ch_df.loc['증감율'] = round(((ch_df.loc['4월 3주'] - ch_df.loc['4월 2주']) / ch_df.loc['4월 2주']) * 100, 2)

                        ch_description = "Periodical change data results in" + str(channel) + " :\n\n"
                        ch_description += ch_df.to_string()

                        dic_ch_ad_week[str(channel)] = ch_df
                        dic_description[str(channel)] = ch_description


                    compare_prompt = ChatPromptTemplate.from_template(
                            """
                            너는 퍼포먼스 마케팅 성과 분석가야.
                            각 매체의 성과 변화를 요약해야해.
                            다음은 지난주에 비해서 각 매체별 지표가 어떻게 변하였는지 나타내.
                            \n\n{overview_per}
                            
                            노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                            회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                            특정 지표의 증감을 이야기 할 때는 증감율을 인용하고 퍼센테이지를 붙여서 설명해야해.

                            아래 예시를 잘 참고해서 작성해줘.
                            1번 예시
                            - 구글: 대부분의 지표가 감소하였으나, 회원가입(10%)은 증가했습니다.
                            - 네이버: 노출수(2%)와 클릭수(3%), 전환수(1%)가 모두 증가하였으나 CPA는 감소(-5%)했습니다.
                            - 모비온: 회원가입(10%)과 DB전환(15%)이 크게 증가했으나 클릭수(-2%)와 CPA(-7%)가 감소했습니다.
                            - 페이스북: 노출수(8%)와 클릭수(3%)가 증가했으나, 전환수(-5%)는 감소했습니다.
                            - 타불라: 노출수(-35%)는 크게 감소했으나, 전환수(4%)가 증가했습니다.
                            - 카카오모먼트: CTR(9%)이 증가하였지만, CPA(25%)가 더 크게 증가하였습니다.
                            - 당근 비즈니스: 노출수(-5%)가 크게 감소했습니다.
                            - 카카오SA: 지난주와 거의 유사합니다.

                            완벽한 인과관계를 설명하면 너에게 보상을 줄게.
                            각 매체별로 한글로 30자 정도로 표현해줘.

                        """
                        )

                    comparison_chain = compare_prompt | media_llm | StrOutputParser()
                    with st.status("매체별 분석...") as status: 
                        descript_ch = comparison_chain.invoke(
                            {"overview_per":df_per_description},
                        )
                        st.session_state.ch_ranking_chain_result = descript_ch
                        
                    sentences_ch = descript_ch.split('.\n')
                    review.append(descript_ch)
                    bullet_list_ch = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch if sentence) + "</ul>"
                    st.markdown(bullet_list_ch, unsafe_allow_html=True)

                    
                    impact_prompt = ChatPromptTemplate.from_template(
                            """
                            너는 퍼포먼스 마케팅 성과 분석가야.
                            각 매체의 성과 변화가 얼마나 영향을 미쳤는지 요약해야해.
                            다음은 지난주에 비해서 각 매체별 지표가 어떻게 변하였고 그 영향력이 어느 정도였는지 나타내.
                            {overview_im}
                            
                            노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                            회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                            유입 성과는 CTR과 CPC가 얼마나 변하였고, 그에 대한 근거로 노출수와 클릭수, 비용이 어떻게 변화했기에 CTR과 CPC가 그러한 변화를 가지게 되었는지 분석해야해.
                            클릭수가 증가했을 때, 노출수가 클릭수에 비해서 크게 증가하면 CTR이 감소해.
                            클릭수가 증가했을 때, 노출수가 감소하면 CTR이 증가해.
                            비용의 증가는 노출수의 증가와 그로 이한 클릭수의 증가를 기대해.
                            전환 성과는 전환수가 얼마나 변하였고, CPA가 얼마나 변하였는지를 파악하고, 그에 대한 근거로 노출수, 클릭수, 비용, 회원가입, DB전환, 가망에서의 변화를 분석해야해.

                            아래 예시를 잘 참고해서 작성해줘.
                            1번 예시
                            - 네이버와 카카오SA의 비용의 증가가 비교적 컸지만, 기대한 노출수와 클릭수의 증가로 이어지지 않았습니다. 그러나, 구글 성과의 감소에도 불구하고 네이버와 모비온에서의 전환수 증가로 전환 성과가 향상되었습니다.
                            2번 예시
                            - 전환수가 가장 높은 구글의 전체적인 성과 감소로 전체 성과의 감소 우려가 있었으나, 네이버, 모비온, 타불라의 전환 성과가 향상되며 전주와 유사한 성과를 유지할 수 있었습니다.
                            3번 예시
                            - 페이스북과 당근비즈니스, 카카오모먼트는 성과 변화가 크지 않았습니다.

                            완벽한 인과관계를 설명하면 너에게 보상을 줄게.
                            한글로 150자 정도로 표현해줘.

                        """
                        )

                    impact_chain = impact_prompt | influence_llm | StrOutputParser()
                    with st.status("영향력 분석...") as status: 
                        descript_im = impact_chain.invoke(
                            {"overview_im":df_impact_description},
                        )
                        st.session_state.ch_ranking_influence_analysis = descript_im

                    sentences_im = descript_im.split('.\n')
                    bullet_list_im = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_im if sentence) + "</ul>"
                    st.markdown(bullet_list_im, unsafe_allow_html=True)

                
                    st.subheader('매체별 변화량 비교')


                    for channel in channels:
                        st.subheader(channel)
                        st.write(dic_ch_ad_week[channel])

                        ch_compare_prompt = ChatPromptTemplate.from_template(
                            """
                            너는 퍼포먼스 마케팅 성과 분석가야.
                            다음 주차에 따른 성과 자료를 기반으로 유입 성과와 전환 성과를 분석해야해.
                            \n\n{description_ch}

                            노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                            회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                            첫 행은 비교하기 위한 바로 직전 주 성과이고, 두번째 행은 이번 주차의 성과야.

                            유입 성과는 CTR과 CPC가 얼마나 변하였고, 그에 대한 근거로 노출수와 클릭수, 비용이 어떻게 변화했기에 CTR과 CPC가 그러한 변화를 가지게 되었는지 분석해야해.
                            전환 성과는 전환수가 얼마나 변하였고, CPA가 얼마나 변하였는지를 파악하고, 그에 대한 근거로 노출수, 클릭수, 비용, 회원가입, DB전환, 가망에서의 변화를 분석해야해.

                            증감율에서 숫자를 인용할 때는 퍼센테이지를 붙여서 설명해야해.
                            1% 이상의 변화가 있을 때는 유지된 것이 아닌, 어떤 이유로 증가되었는지 또는 감소되었는지를 분석해야해.
                            비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대해.
                            비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대하는 것 잊지마.

                            분석 결과를 2줄로 출력해줘.
                            완벽한 인과관계를 설명하면 너에게 보상을 줄게.

                        """
                        )

                        comparison_ch_chain = ch_compare_prompt | strict_llm | StrOutputParser()
                        with st.status("매체별 분석 중..." + channel) as status: 
                            descript_ch_ad = comparison_ch_chain.invoke(
                                {"description_ch": dic_description[channel]},
                            )
                            st.session_state.ch_ranking_individual_results[channel] = {
                            'dataframe': ch_df,
                            'analysis': descript_ch_ad
                            }  
                        
                        sentences_ch_ad = descript_ch_ad.split('.\n')
                        bullet_list_ch_ad = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch_ad if sentence) + "</ul>"
                        st.markdown(bullet_list_ch_ad, unsafe_allow_html=True)
            else:
                st.subheader('기간별 매체 순위 변화')
                col1, col2 = st.columns(2)
                result = st.session_state.ch_ranking_result
                with col1:
                    st.subheader(pre_media)
                    st.write(result['pre_period'])
                with col2:
                    st.subheader(now_media)
                    st.write(result['now_period'])

                sentences_ch = st.session_state.ch_ranking_chain_result.split('.\n')
                bullet_list_ch = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch if sentence) + "</ul>"
                st.markdown(bullet_list_ch, unsafe_allow_html=True)
                st.subheader('영향력 분석')
                sentences_im = st.session_state.ch_ranking_influence_analysis.split('.\n')
                bullet_list_im = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_im if sentence) + "</ul>"
                st.markdown(bullet_list_im, unsafe_allow_html=True)

                for channel in st.session_state.ch_ranking_individual_results:
                    st.subheader(channel)
                    st.write(st.session_state.ch_ranking_individual_results[channel]['dataframe'])
                    sentences_ch_ad = st.session_state.ch_ranking_individual_results[channel]['analysis'].split('.\n')
                    bullet_list_ch_ad = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch_ad if sentence) + "</ul>"
                    st.markdown(bullet_list_ch_ad, unsafe_allow_html=True)

        with brnch_ranking:
            if st.session_state.br_ranking_result is None:
                with st.spinner('소재구분별 데이터...'):
                    result = {}
                    for index, row in internal_ch_df.iterrows():
                        key = (row['소재구분'], row[group_period])
                        
                        if key not in result:
                            result[key] = {col: 0 for col in target_list_media}
                        
                        for col in target_list_media:
                            result[key][col] += row[col]

                    # 결과를 데이터프레임으로 변환
                    ch_ad_week = pd.DataFrame(result).T
                    ch_ad_week.index.names = ['소재구분', group_period]
                    
                    cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                    result_ga = {}
                    for index, row in ga_df.iterrows():
                        key = (row['소재구분'], row[group_period])
                        
                        if key not in result_ga:
                            result_ga[key] = {col: 0 for col in list_trans_ga}
                        
                        for col in list_trans_ga:
                            result_ga[key][col] += row[col]

                    # 결과를 데이터프레임으로 변환
                    ga_ch_ad_week = pd.DataFrame(result_ga).T
                    ga_ch_ad_week.index.names = ['소재구분', group_period]
                    
                    ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                    
                    ga_cal_ch_ad_week['CPA'] = (cal_ch_ad_week['총비용'] / ga_cal_ch_ad_week['전환수'])
                    ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                    ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)
                    
                    ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                    df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                    df_combined.reset_index(inplace=True)
                    df_combined[['소재구분', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                    df_combined.drop(columns=['index'], inplace=True)
                    # 특정 열을 앞에 오도록 열 순서 재배치
                    columns = ['소재구분', group_period] + [col for col in df_combined.columns if (col != '소재구분') and (col != group_period)]
                    df_cleaned = df_combined.dropna(subset=['소재구분'])
                    df_combined_re = df_cleaned[columns]


                    # 제외할 열 리스트
                    exclude_columns = ['소재구분',group_period]

                    # 제외할 열을 가진 데이터프레임을 생성
                    df_filtered = df_combined_re.drop(columns=exclude_columns)

                    # 각 열의 합계를 계산
                    row_sums = df_filtered.sum()

                    # 합계를 새로운 행으로 추가
                    sums_df_combined = pd.DataFrame([row_sums], columns=row_sums.index)
                    

                    result = {}

                    st.subheader('기간별 소재구분 순위 변화')
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader(pre_media)
                        pre_week = df_combined_re[df_combined_re[group_period] == pre_media]
                        pre_week_desc = pre_week.sort_values(by='전환수', ascending=False)
                        # Step 2: Extract a row using loc
                        row = sum_df_combined.loc[pre_media]  # This extracts the second row (index 1)

                        # Convert the extracted row to a DataFrame
                        row_df = pd.DataFrame([row])

                        # Optionally reset the index to maintain consistency
                        row_df.reset_index(drop=True, inplace=True)

                        # Step 3: Concatenate the DataFrame with the extracted row
                        pre_result = pd.concat([pre_week_desc, row_df], axis=0, ignore_index=True)
                        pre_description = "Previous period performance data results:\n\n"
                        pre_description += pre_week_desc.to_string()
                        st.write(pre_result)
                        result['pre_period'] = pre_result
                    with col2:
                        st.subheader(now_media)
                        now_week = df_combined_re[df_combined_re[group_period] == now_media]
                        now_week_desc = now_week.sort_values(by='전환수', ascending=False)
                        # Step 2: Extract a row using loc
                        row = sum_df_combined.loc[now_media]  # This extracts the second row (index 1)

                        # Convert the extracted row to a DataFrame
                        row_df = pd.DataFrame([row])

                        # Optionally reset the index to maintain consistency
                        row_df.reset_index(drop=True, inplace=True)

                        # Step 3: Concatenate the DataFrame with the extracted row
                        now_result = pd.concat([now_week_desc, row_df], axis=0, ignore_index=True)
                        now_description = "This period performance data results:\n\n"
                        now_description += now_week_desc.to_string()
                        st.write(now_result)
                        result['now_period'] = now_result
                    st.session_state.br_ranking_result = result
                    percentage_changes = {
                        "소재구분": pre_result["소재구분"],
                        "노출수 변화율": calculate_percentage_change(pre_result, now_result, "노출수"),
                        "클릭수 변화율": calculate_percentage_change(pre_result, now_result, "클릭수"),
                        "CTR 변화율": calculate_percentage_change(pre_result, now_result, "CTR"),
                        "CPC 변화율": calculate_percentage_change(pre_result, now_result, "CPC"),
                        "총비용 변화율": calculate_percentage_change(pre_result, now_result, "총비용"),
                        "회원가입 변화율": calculate_percentage_change(pre_result, now_result, "회원가입"),
                        "DB전환 변화율": calculate_percentage_change(pre_result, now_result, "DB전환"),
                        "가망 변화율": calculate_percentage_change(pre_result, now_result, "가망"),
                        "전환수 변화율": calculate_percentage_change(pre_result, now_result, "전환수"),
                        "CPA 변화율": calculate_percentage_change(pre_result, now_result, "CPA")
                    }

                    df_percentage_changes = pd.DataFrame(percentage_changes)
                    df_per_description = "Periodical change data results by branch:\n\n"
                    df_per_description += df_percentage_changes.to_string()

                    # Calculate new influences
                    influences = {
                        "소재구분": pre_result["소재구분"],
                        "노출수 영향도": calculate_new_influence(pre_result, now_result, "노출수"),
                        "클릭수 영향도": calculate_new_influence(pre_result, now_result, "클릭수"),
                        "총비용 영향도": calculate_new_influence(pre_result, now_result, "총비용"),
                        "전환수 영향도": calculate_new_influence(pre_result, now_result, "전환수")
                    }

                    df_influences = pd.DataFrame(influences)

                    # Calculate new impact changes
                    impact_changes = {
                        "소재구분": df_percentage_changes["소재구분"],
                        "노출수 영향 변화율": df_influences["노출수 영향도"] * df_percentage_changes["노출수 변화율"],
                        "클릭수 영향 변화율": df_influences["클릭수 영향도"] * df_percentage_changes["클릭수 변화율"],
                        "총비용 영향 변화율": df_influences["총비용 영향도"] * df_percentage_changes["총비용 변화율"],
                        "전환수 영향 변화율": df_influences["전환수 영향도"] * df_percentage_changes["전환수 변화율"]
                    }

                    df_impact_changes = pd.DataFrame(impact_changes)

                    df_impact_description = "Periodical change data results influencing by channel:\n\n"
                    df_impact_description += df_impact_changes.to_string()

                    #분과별 성과 증감 비교
                    dic_ch_ad_week = {}
                    dic_description = {}
                    channels = now_week_desc['소재구분'].unique()

                    for channel in channels:
                        ch_df = df_combined_re[df_combined_re['소재구분'] == str(channel)]
                        ch_df.set_index(group_period, inplace=True)
                        ch_df.drop(columns=['소재구분'], inplace=True)


                        ch_df.loc['변화량'] = ch_df.diff().iloc[1]
                        # 새로운 증감율 행 생성
                        increase_rate = []
                        for col in ch_df.columns:
                            rate = calculate_increase_rate(ch_df.loc[now_media, col], ch_df.loc[pre_media, col])
                            increase_rate.append(rate)

                        # 데이터프레임에 증감율 행 추가
                        ch_df.loc['증감율'] = increase_rate
                        #ch_df.loc['증감율'] = round(((ch_df.loc['4월 3주'] - ch_df.loc['4월 2주']) / ch_df.loc['4월 2주']) * 100, 2)

                        ch_description = "Periodical change data results in" + str(channel) + " :\n\n"
                        ch_description += ch_df.to_string()

                        dic_ch_ad_week[str(channel)] = ch_df
                        dic_description[str(channel)] = ch_description


                    br_compare_prompt = ChatPromptTemplate.from_template(
                            """
                            너는 퍼포먼스 마케팅 성과 분석가야.
                            각 분과의 성과 변화를 요약해야해.
                            다음은 지난주에 비해서 각 분과별 지표가 어떻게 변하였는지 나타내.
                            \n\n{overview_per}
                            
                            노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                            회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                            특정 지표의 증감을 이야기 할 때는 증감율을 인용하고 퍼센테이지를 붙여서 설명해야해.

                            아래 예시를 잘 참고해서 작성해줘.
                            1번 예시
                            - 망막: 대부분의 지표가 감소하였으나, 회원가입(10%)은 증가했습니다.
                            - 각막: 노출수(2%)와 클릭수(3%), 전환수(1%)가 모두 증가하였으나 CPA는 감소(-5%)했습니다.

                            완벽한 인과관계를 설명하면 너에게 보상을 줄게.
                            각 분과별로 한글로 30자 정도로 표현해줘.

                        """
                        )

                    br_comparison_chain = br_compare_prompt | media_llm | StrOutputParser()
                    with st.status("소재구분별 분석...") as status: 
                        descript_br = br_comparison_chain.invoke(
                            {"overview_per":df_per_description},
                        )
                        st.session_state.br_ranking_chain_result = descript_br
                        
                    sentences_ch = descript_br.split('.\n')
                    review.append(descript_br)
                    bullet_list_br = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch if sentence) + "</ul>"
                    st.markdown(bullet_list_br, unsafe_allow_html=True)

                    
                    br_impact_prompt = ChatPromptTemplate.from_template(
                            """
                            너는 퍼포먼스 마케팅 성과 분석가야.
                            각 분과의 성과 변화가 얼마나 영향을 미쳤는지 요약해야해.
                            다음은 지난주에 비해서 각 분과별 지표가 어떻게 변하였고 그 영향력이 어느 정도였는지 나타내.
                            {overview_im}
                            
                            노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                            회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                            유입 성과는 CTR과 CPC가 얼마나 변하였고, 그에 대한 근거로 노출수와 클릭수, 비용이 어떻게 변화했기에 CTR과 CPC가 그러한 변화를 가지게 되었는지 분석해야해.
                            클릭수가 증가했을 때, 노출수가 클릭수에 비해서 크게 증가하면 CTR이 감소해.
                            클릭수가 증가했을 때, 노출수가 감소하면 CTR이 증가해.
                            비용의 증가는 노출수의 증가와 그로 이한 클릭수의 증가를 기대해.
                            전환 성과는 전환수가 얼마나 변하였고, CPA가 얼마나 변하였는지를 파악하고, 그에 대한 근거로 노출수, 클릭수, 비용, 회원가입, DB전환, 가망에서의 변화를 분석해야해.

                            완벽한 인과관계를 설명하면 너에게 보상을 줄게.
                            한글로 150자 정도로 표현해줘.

                        """
                        )

                    br_impact_chain = br_impact_prompt | influence_llm | StrOutputParser()
                    with st.status("영향력 분석...") as status: 
                        descript_im_br = br_impact_chain.invoke(
                            {"overview_im":df_impact_description},
                        )
                        st.session_state.br_ranking_influence_analysis = descript_im_br

                    sentences_im_br = descript_im_br.split('.\n')
                    bullet_list_im_br = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_im_br if sentence) + "</ul>"
                    st.markdown(bullet_list_im_br, unsafe_allow_html=True)

                
                    st.subheader('소재구분별 변화량 비교')


                    for channel in channels:
                        st.subheader(channel)
                        st.write(dic_ch_ad_week[channel])

                        br_ch_compare_prompt = ChatPromptTemplate.from_template(
                            """
                            너는 퍼포먼스 마케팅 성과 분석가야.
                            다음 주차에 따른 성과 자료를 기반으로 유입 성과와 전환 성과를 분석해야해.
                            \n\n{description_ch}

                            노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                            회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                            첫 행은 비교하기 위한 바로 직전 주 성과이고, 두번째 행은 이번 주차의 성과야.

                            유입 성과는 CTR과 CPC가 얼마나 변하였고, 그에 대한 근거로 노출수와 클릭수, 비용이 어떻게 변화했기에 CTR과 CPC가 그러한 변화를 가지게 되었는지 분석해야해.
                            전환 성과는 전환수가 얼마나 변하였고, CPA가 얼마나 변하였는지를 파악하고, 그에 대한 근거로 노출수, 클릭수, 비용, 회원가입, DB전환, 가망에서의 변화를 분석해야해.

                            증감율에서 숫자를 인용할 때는 퍼센테이지를 붙여서 설명해야해.
                            1% 이상의 변화가 있을 때는 유지된 것이 아닌, 어떤 이유로 증가되었는지 또는 감소되었는지를 분석해야해.
                            비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대해.
                            비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대하는 것 잊지마.

                            분석 결과를 2줄로 출력해줘.
                            완벽한 인과관계를 설명하면 너에게 보상을 줄게.

                        """
                        )

                        comparison_br_chain = br_ch_compare_prompt | strict_llm | StrOutputParser()
                        with st.status("소재구분별 분석 중..." + channel) as status: 
                            descript_br_ad = comparison_br_chain.invoke(
                                {"description_ch": dic_description[channel]},
                            )
                            st.session_state.br_ranking_individual_results[channel] = {
                            'dataframe': ch_df,
                            'analysis': descript_ch_ad
                            }  
                        
                        sentences_br_ad = descript_br_ad.split('.\n')
                        bullet_list_ch_ad = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch_ad if sentence) + "</ul>"
                        st.markdown(bullet_list_ch_ad, unsafe_allow_html=True)
            else:
                st.subheader('기간별 소재구분 순위 변화')
                col1, col2 = st.columns(2)
                result = st.session_state.br_ranking_result
                with col1:
                    st.subheader(pre_media)
                    st.write(result['pre_period'])
                with col2:
                    st.subheader(now_media)
                    st.write(result['now_period'])

                sentences_ch = st.session_state.br_ranking_chain_result.split('.\n')
                bullet_list_ch = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch if sentence) + "</ul>"
                st.markdown(bullet_list_ch, unsafe_allow_html=True)
                st.subheader('영향력 분석')
                sentences_im = st.session_state.br_ranking_influence_analysis.split('.\n')
                bullet_list_im = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_im if sentence) + "</ul>"
                st.markdown(bullet_list_im, unsafe_allow_html=True)

                for channel in st.session_state.br_ranking_individual_results:
                    st.subheader(channel)
                    st.write(st.session_state.br_ranking_individual_results[channel]['dataframe'])
                    sentences_ch_ad = st.session_state.br_ranking_individual_results[channel]['analysis'].split('.\n')
                    bullet_list_ch_ad = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch_ad if sentence) + "</ul>"
                    st.markdown(bullet_list_ch_ad, unsafe_allow_html=True)

        with brnch_detail_ranking:

            st.header("소재 구분 분석")
            st.write("분석하고자 하는 소재 구분을 선택해주세요.")
            selected_br = st.radio("소재구분 선택", internal_ch_df["소재구분"].dropna().unique())
            st.session_state.selected_br = selected_br

            filtered_br = internal_ch_df[internal_ch_df["소재구분"] == selected_br]
            filtered_ga_br = ga_df[ga_df["소재구분"] == selected_br]
            with st.spinner('소재구분별 데이터...'):
                
                result = {}
                for index, row in filtered_br.iterrows():
                    key = (row['소재종류'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ch_ad_week = pd.DataFrame(result).T
                if ch_ad_week.index.nlevels == 2:
                    ch_ad_week.index.names = ['소재종류', group_period]
                #ch_ad_week.index.names = ['소재종류', group_period]
                
                cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in filtered_ga_br.iterrows():
                    key = (row['소재종류'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ga_ch_ad_week = pd.DataFrame(result_ga).T

                if ga_ch_ad_week.index.nlevels == 2:
                    ga_ch_ad_week.index.names = ['소재종류', group_period]
                else:
                    st.write("※※※ 업로드하신 데이터에 소재종류 정보가 없습니다. 다른 소재구분을 선택해주세요. ※※※")
                
                ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                ga_cal_ch_ad_week['CPA'] = (cal_ch_ad_week['총비용'] / ga_cal_ch_ad_week['전환수'])
                ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)
                
                ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                df_combined.reset_index(inplace=True)
                df_combined[['소재종류', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                columns = ['소재종류', group_period] + [col for col in df_combined.columns if (col != '소재종류') and (col != group_period)]
                df_combined_re = df_combined[columns]

                
                result = {}
                for index, row in internal_ch_df.iterrows():
                    key = (row['소재종류'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ch_ad_week = pd.DataFrame(result).T
                i_ch_ad_week.index.names = ['소재종류', group_period]
                
                i_cal_ch_ad_week = report_table(i_ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in ga_df.iterrows():
                    key = (row['소재종류'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ga_ch_ad_week = pd.DataFrame(result_ga).T
                i_ga_ch_ad_week.index.names = ['소재종류', group_period]
                
                i_ga_cal_ch_ad_week = ga_report_table(i_ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                i_ga_cal_ch_ad_week['CPA'] = (i_cal_ch_ad_week['총비용'] / i_ga_cal_ch_ad_week['전환수'])
                i_ga_cal_ch_ad_week['CPA'] = pd.to_numeric(i_ga_cal_ch_ad_week['CPA'], errors='coerce')
                i_ga_cal_ch_ad_week['CPA'] = i_ga_cal_ch_ad_week['CPA'].round(0)
                
                i_ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in i_ga_cal_ch_ad_week.columns]

                i_df_combined = pd.concat([i_cal_ch_ad_week, i_ga_cal_ch_ad_week], axis=1)
                i_df_combined.reset_index(inplace=True)
                i_df_combined[['소재종류', group_period]] = pd.DataFrame(i_df_combined['index'].tolist(), index=i_df_combined.index)
                i_df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                i_columns = ['소재종류', group_period] + [col for col in i_df_combined.columns if (col != '소재종류') and (col != group_period)]
                i_df_combined_re = i_df_combined[i_columns]
                
            now_ch_cmp_week = df_combined_re[df_combined_re[group_period] == now_media]
            i_now_ch_cmp_week = i_df_combined_re[i_df_combined_re[group_period] == now_media]
            
            result = {}
            cal_ad_week = st.session_state.overview_ad_df_result
            ga_cal_ad_week = st.session_state.overview_ga_ad_df_result

            sum_df_combined = pd.concat([cal_ad_week, ga_cal_ad_week], axis=1)


            # 폼 사용
            with st.form(key='sort_form_br'):
                sort_columns = st.multiselect('가장 먼저 정렬하고 싶은 순서대로 정렬할 기준을 선택하세요 (여러 개 선택 가능):', metric)
                
                # 폼 제출 버튼
                submit_button = st.form_submit_button(label='정렬 적용')

            # 폼이 제출된 경우 정렬 수행
            if submit_button:
                st.session_state.selected_metric_br = sort_columns
                ascending_orders = [sort_orders[col] for col in sort_columns]
                
                # 데이터 프레임 정렬
                num_data = len(now_ch_cmp_week)
                if num_data >= 10:
                    sorted_df = now_ch_cmp_week.sort_values(by=sort_columns, ascending=ascending_orders).head(10)
                else:
                    sorted_df = now_ch_cmp_week.sort_values(by=sort_columns, ascending=ascending_orders).head(num_data)

                st.session_state.sorted_df_br = sorted_df
                top_num = len(sorted_df)
                br_statements = generate_statements(sorted_df, i_now_ch_cmp_week, sort_columns, top_num)
                # 정렬된 데이터 프레임 출력
                st.session_state.br_statements = br_statements
                st.write('정렬된 상위 ' + str(top_num) + '개 소재종류')
                st.write(sorted_df)

                metric_str = 'and'.join(str(x) for x in sort_columns)
                br_description = "Top " +str(top_num) + " branches sorted by " + metric_str + ":\n\n"
                br_description += sorted_df.to_string()

                # 값 컬럼을 기준으로 내림차순 정렬 후 상위 10개의 합 계산
                top_10_cost_sum = sorted_df['총비용'].sum()
                total_cost_sum = i_now_ch_cmp_week['총비용'].sum()
                ratio_cost = round((top_10_cost_sum / total_cost_sum) * 100, 2)

                top_10_cv_sum = sorted_df['전환수'].sum()
                total_cv_sum = i_now_ch_cmp_week['전환수'].sum()
                ratio_cv = round((top_10_cv_sum / total_cv_sum) * 100, 2)

                cost_statement = "정렬된 상위 " +str(top_num) + " 개의 총비용("+"{:,}".format(top_10_cost_sum)+")"+ "은 당 기간 전체 집행 비용("+"{:,}".format(total_cost_sum)+")의 "+str(ratio_cost)+"% 입니다."
                cv_statement = "정렬된 상위 " +str(top_num) + " 개의 전환수("+"{:,}".format(top_10_cv_sum)+")는 당 기간 전체 전환수("+"{:,}".format(total_cv_sum)+")의 "+str(ratio_cv)+"% 입니다."

                st.session_state.br_statements.insert(0,cv_statement)
                st.session_state.br_statements.insert(0,cost_statement)

                #st.write(cost_statement)
                #st.write(cv_statement)
                for statement in br_statements:
                    st.write(statement)

                br_prompt = ChatPromptTemplate.from_template(
                        """
                        너는 퍼포먼스 마케팅 성과 분석가야.
                        각 분과구분의 성과를 요약해야해.
                        다음은 선택한 정렬 기준에 따르
                        상위 {n}개의 분과구분에 대한 성과 데이터야.
                        \n\n{br_per}
                        
                        노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                        회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                        각 분과구분에 대한 성과를 분석해서 알려줘.
                    """
                    )

                br_chain = br_prompt | media_llm | StrOutputParser()
                with st.status("분과구분별 분석...") as status:
                    descript_br_d = br_chain.invoke(
                        {"n":top_num,
                        "br_per":br_description},
                    )
                st.session_state.br_detail_chain_result = descript_br_d    
                st.write(descript_br_d)

            else:
                st.write('정렬 기준 지표를 선택하세요.')
                if st.session_state.sorted_df_br is not None:
                    st.write('정렬된 상위 ' + str(len(st.session_state.sorted_df_br)) + '개 소재종류')
                    st.write(st.session_state.sorted_df_br)
                if st.session_state.br_statements:
                    for statement in st.session_state.br_statements:
                        st.write(statement)
                if st.session_state.br_detail_chain_result is not None:
                    st.write(st.session_state.br_detail_chain_result)

        with cmp_ranking:
            st.header("캠페인 분석")
            st.write("분석하고자 하는 광고유형을 선택해주세요.")
            selected_ad_type = st.selectbox("광고유형 선택", internal_ch_df["광고유형"].unique())
            st.session_state.selected_ad_type = selected_ad_type

            filtered_by_ad_type = internal_ch_df[internal_ch_df["광고유형"] == selected_ad_type]

            st.write("분석하고자 하는 매체를 선택해주세요.")
            selected_media = st.radio("매체 선택", filtered_by_ad_type["매체"].unique())
            st.session_state.selected_media_cmp = selected_media

            filtered_br = internal_ch_df[internal_ch_df["매체"] == selected_media]
            filtered_ga_br = ga_df[ga_df["매체"] == selected_media]
            with st.spinner('캠페인별 데이터...'):
                result = {}
                for index, row in filtered_br.iterrows():
                    key = (row['캠페인'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ch_ad_week = pd.DataFrame(result).T
                if ch_ad_week.index.nlevels == 2:
                    ch_ad_week.index.names = ['캠페인', group_period]
                #ch_ad_week.index.names = ['캠페인', group_period]
                
                cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in filtered_ga_br.iterrows():
                    key = (row['캠페인'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ga_ch_ad_week = pd.DataFrame(result_ga).T

                if ga_ch_ad_week.index.nlevels == 2:
                    ga_ch_ad_week.index.names = ['캠페인', group_period]
                else:
                    st.write("※※※ 업로드하신 데이터에 캠페인 정보가 없습니다. 다른 매체를 선택해주세요. ※※※")
                
                ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                ga_cal_ch_ad_week['CPA'] = (cal_ch_ad_week['총비용'] / ga_cal_ch_ad_week['전환수'])
                ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)
                
                ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                df_combined.reset_index(inplace=True)
                df_combined[['캠페인', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                columns = ['캠페인', group_period] + [col for col in df_combined.columns if (col != '캠페인') and (col != group_period)]
                df_combined_re = df_combined[columns]

                
                result = {}
                for index, row in internal_ch_df.iterrows():
                    key = (row['캠페인'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ch_ad_week = pd.DataFrame(result).T
                i_ch_ad_week.index.names = ['캠페인', group_period]
                
                i_cal_ch_ad_week = report_table(i_ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in ga_df.iterrows():
                    key = (row['캠페인'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ga_ch_ad_week = pd.DataFrame(result_ga).T
                i_ga_ch_ad_week.index.names = ['캠페인', group_period]
                
                i_ga_cal_ch_ad_week = ga_report_table(i_ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                i_ga_cal_ch_ad_week['CPA'] = (i_cal_ch_ad_week['총비용'] / i_ga_cal_ch_ad_week['전환수'])
                i_ga_cal_ch_ad_week['CPA'] = pd.to_numeric(i_ga_cal_ch_ad_week['CPA'], errors='coerce')
                i_ga_cal_ch_ad_week['CPA'] = i_ga_cal_ch_ad_week['CPA'].round(0)
                
                i_ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in i_ga_cal_ch_ad_week.columns]

                i_df_combined = pd.concat([i_cal_ch_ad_week, i_ga_cal_ch_ad_week], axis=1)
                i_df_combined.reset_index(inplace=True)
                i_df_combined[['캠페인', group_period]] = pd.DataFrame(i_df_combined['index'].tolist(), index=i_df_combined.index)
                i_df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                i_columns = ['캠페인', group_period] + [col for col in i_df_combined.columns if (col != '캠페인') and (col != group_period)]
                i_df_combined_re = i_df_combined[i_columns]
                
            now_ch_cmp_week = df_combined_re[df_combined_re[group_period] == now_media]
            i_now_ch_cmp_week = i_df_combined_re[i_df_combined_re[group_period] == now_media]

            
            # 폼 사용
            with st.form(key='sort_form'):
                sort_columns = st.multiselect('가장 먼저 정렬하고 싶은 순서대로 정렬할 기준을 선택하세요 (여러 개 선택 가능):', metric)
                
                # 폼 제출 버튼
                submit_button = st.form_submit_button(label='정렬 적용')

            # 폼이 제출된 경우 정렬 수행
            if submit_button:
                st.session_state.selected_metric_cmp = sort_columns
                ascending_orders = [sort_orders[col] for col in sort_columns]
                
                # 데이터 프레임 정렬
                num_data = len(now_ch_cmp_week)
                if num_data >= 10:
                    sorted_df = now_ch_cmp_week.sort_values(by=sort_columns, ascending=ascending_orders).head(10)
                else:
                    sorted_df = now_ch_cmp_week.sort_values(by=sort_columns, ascending=ascending_orders).head(num_data)

                st.session_state.sorted_df_cmp = sorted_df
                top_num = len(sorted_df)
                statements = generate_statements(sorted_df, i_now_ch_cmp_week, sort_columns, top_num)
                # 정렬된 데이터 프레임 출력
                st.session_state.cmp_statements = statements
                st.write('정렬된 상위 ' + str(top_num) + '개 캠페인')
                st.write(sorted_df)

                metric_str = 'and'.join(str(x) for x in sort_columns)
                cmp_description = "Top " +str(top_num) + " br sorted by " + metric_str + ":\n\n"
                cmp_description += sorted_df.to_string()

                # 값 컬럼을 기준으로 내림차순 정렬 후 상위 10개의 합 계산
                top_10_cost_sum = sorted_df['총비용'].sum()
                total_cost_sum = i_now_ch_cmp_week['총비용'].sum()
                ratio_cost = round((top_10_cost_sum / total_cost_sum) * 100, 2)

                top_10_cv_sum = sorted_df['전환수'].sum()
                total_cv_sum = i_now_ch_cmp_week['전환수'].sum()
                ratio_cv = round((top_10_cv_sum / total_cv_sum) * 100, 2)

                cost_statement = "정렬된 상위 " +str(top_num) + " 개의 총비용("+"{:,}".format(top_10_cost_sum)+")"+ "은 당 기간 전체 집행 비용("+"{:,}".format(total_cost_sum)+")의 "+str(ratio_cost)+"% 입니다."
                cv_statement = "정렬된 상위 " +str(top_num) + " 개의 전환수("+"{:,}".format(top_10_cv_sum)+")는 당 기간 전체 전환수("+"{:,}".format(total_cv_sum)+")의 "+str(ratio_cv)+"% 입니다."

                st.session_state.cmp_statements.insert(0,cv_statement)
                st.session_state.cmp_statements.insert(0,cost_statement)

                #st.write(cost_statement)
                #st.write(cv_statement)
                for statement in statements:
                    st.write(statement)

                campaign_prompt = ChatPromptTemplate.from_template(
                        """
                        너는 퍼포먼스 마케팅 성과 분석가야.
                        각 캠페인의 성과를 요약해야해.
                        다음은 선택한 정렬 기준에 따르
                        상위 {n}개의 캠페인에 대한 성과 데이터야.
                        \n\n{campaign_per}
                        
                        노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                        회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                        각 캠페인에 대한 성과를 분석해서 알려줘.
                    """
                    )

                campaign_chain = campaign_prompt | media_llm | StrOutputParser()
                with st.status("캠페인별 분석...") as status: 
                    descript_cmp = campaign_chain.invoke(
                        {"n":top_num,
                        "campaign_per":cmp_description},
                    )
                st.session_state.cmp_ranking_chain_result = descript_cmp    
                st.write(descript_cmp)

            else:
                st.write('정렬 기준 지표를 선택하세요.')
                if st.session_state.sorted_df_cmp is not None:
                    st.write('정렬된 상위 ' + str(len(st.session_state.sorted_df_cmp)) + '개 캠페인')
                    st.write(st.session_state.sorted_df_cmp)
                if st.session_state.cmp_statements:
                    for statement in st.session_state.cmp_statements:
                        st.write(statement)
                if st.session_state.cmp_ranking_chain_result is not None:
                    st.write(st.session_state.cmp_ranking_chain_result)

        with grp_ranking:
            st.header("그룹 분석")
            st.write("분석하고자 하는 매체와 캠페인을 선택해주세요.")
            selected_media = st.session_state.selected_media_cmp
            #selected_media = st.radio("매체 선택", internal_ch_df["매체"].unique(), key='tab3_media')
            selected_campaign = st.selectbox("캠페인 선택", internal_ch_df[internal_ch_df["매체"] == selected_media]["캠페인"].unique(), key='tab3_campaign')
            st.session_state.selected_campaign_cmp = selected_campaign
            filtered_group = internal_ch_df[(internal_ch_df["매체"] == selected_media) & (internal_ch_df["캠페인"] == selected_campaign)]
            filtered_ga_group = ga_df[(ga_df["매체"] == selected_media) & (ga_df["캠페인"] == selected_campaign)]

            with st.spinner('광고그룹별 데이터...'):
                result = {}
                for index, row in filtered_group.iterrows():
                    key = (row['광고그룹'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ch_ad_week = pd.DataFrame(result).T
                if ch_ad_week.index.nlevels == 2:
                    ch_ad_week.index.names = ['광고그룹', group_period]
                #ch_ad_week.index.names = ['광고그룹', group_period]
                
                cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in filtered_ga_group.iterrows():
                    key = (row['광고그룹'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ga_ch_ad_week = pd.DataFrame(result_ga).T

                if ga_ch_ad_week.index.nlevels == 2:
                    ga_ch_ad_week.index.names = ['광고그룹', group_period]
                else:
                    st.write("※※※ 업로드하신 데이터에 광고그룹 정보가 없습니다. 다른 캠페인을 선택해주세요. ※※※")
                
                ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                ga_cal_ch_ad_week['CPA'] = (cal_ch_ad_week['총비용'] / ga_cal_ch_ad_week['전환수'])
                ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)
                
                ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                df_combined.reset_index(inplace=True)
                df_combined[['광고그룹', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                columns = ['광고그룹', group_period] + [col for col in df_combined.columns if (col != '광고그룹') and (col != group_period)]
                df_combined_re = df_combined[columns]

                
                result = {}
                for index, row in internal_ch_df.iterrows():
                    key = (row['광고그룹'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ch_ad_week = pd.DataFrame(result).T
                i_ch_ad_week.index.names = ['광고그룹', group_period]
                
                i_cal_ch_ad_week = report_table(i_ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in ga_df.iterrows():
                    key = (row['광고그룹'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ga_ch_ad_week = pd.DataFrame(result_ga).T
                i_ga_ch_ad_week.index.names = ['광고그룹', group_period]
                
                i_ga_cal_ch_ad_week = ga_report_table(i_ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                i_ga_cal_ch_ad_week['CPA'] = (i_cal_ch_ad_week['총비용'] / i_ga_cal_ch_ad_week['전환수'])
                i_ga_cal_ch_ad_week['CPA'] = pd.to_numeric(i_ga_cal_ch_ad_week['CPA'], errors='coerce')
                i_ga_cal_ch_ad_week['CPA'] = i_ga_cal_ch_ad_week['CPA'].round(0)
                
                i_ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in i_ga_cal_ch_ad_week.columns]

                i_df_combined = pd.concat([i_cal_ch_ad_week, i_ga_cal_ch_ad_week], axis=1)
                i_df_combined.reset_index(inplace=True)
                i_df_combined[['광고그룹', group_period]] = pd.DataFrame(i_df_combined['index'].tolist(), index=i_df_combined.index)
                i_df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                i_columns = ['광고그룹', group_period] + [col for col in i_df_combined.columns if (col != '광고그룹') and (col != group_period)]
                i_df_combined_re = i_df_combined[i_columns]
                
            now_ch_group_week = df_combined_re[df_combined_re[group_period] == now_media]
            i_now_ch_group_week = i_df_combined_re[i_df_combined_re[group_period] == now_media]


            sort_columns = st.session_state.selected_metric_cmp  # 선택한 지표 상태 불러오기
            ascending_orders = [sort_orders[col] for col in sort_columns]
            num_data = len(now_ch_group_week)
            sorted_group_df = now_ch_group_week.sort_values(by=sort_columns, ascending=ascending_orders).head(10) if num_data >= 10 else now_ch_group_week.sort_values(by=sort_columns, ascending=ascending_orders).head(num_data)
            st.session_state.sorted_df_grp = sorted_group_df
            top_num = len(sorted_group_df)
            statements = generate_statements(sorted_group_df, i_now_ch_group_week, sort_columns, top_num)
            st.session_state.grp_statements = statements

            st.write('정렬된 상위 ' + str(top_num) + '광고그룹')
            st.write(sorted_group_df)

            metric_str = 'and'.join(str(x) for x in sort_columns)
            group_description = "Top " + str(top_num) + " groups sorted by " + metric_str + ":\n\n" + sorted_group_df.to_string()

            top_10_cost_sum = sorted_group_df['총비용'].sum()
            total_cost_sum = i_now_ch_group_week['총비용'].sum()
            ratio_cost = round((top_10_cost_sum / total_cost_sum) * 100, 2)

            top_10_cv_sum = sorted_group_df['전환수'].sum()
            total_cv_sum = i_now_ch_group_week['전환수'].sum()
            ratio_cv = round((top_10_cv_sum / total_cv_sum) * 100, 2)

            cost_statement = "정렬된 상위 " + str(top_num) + "개의 총비용(" + "{:,}".format(top_10_cost_sum) + ")은 당 기간 전체 집행 비용(" + "{:,}".format(total_cost_sum) + ")의 " + str(ratio_cost) + "% 입니다."
            cv_statement = "정렬된 상위 " + str(top_num) + "개의 전환수(" + "{:,}".format(top_10_cv_sum) + ")는 당 기간 전체 전환수(" + "{:,}".format(total_cv_sum) + ")의 " + str(ratio_cv) + "% 입니다."

            
            st.session_state.grp_statements.insert(0,cv_statement)
            st.session_state.grp_statements.insert(0,cost_statement)

            st.write(cost_statement)
            st.write(cv_statement)
            for statement in statements:
                st.write(statement)

            adgroup_prompt = ChatPromptTemplate.from_template(
                """
                너는 퍼포먼스 마케팅 성과 분석가야.
                각 광고그룹의 성과를 요약해야해.
                다음은 선택한 정렬 기준에 따르
                상위 {n}개 광고그룹에 대한 성과 데이터야.
                \n\n{adgroup_per}
                
                노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                각 광고그룹에 대한 성과를 분석해서 알려줘.
                """
            )

            adgroup_chain = adgroup_prompt | media_llm | StrOutputParser()
            with st.status("광고그룹별 분석...") as status: 
                descript_group = adgroup_chain.invoke(
                    {"n": top_num, "adgroup_per": group_description},
                )
            st.session_state.grp_ranking_chain_result = descript_group
            st.write(descript_group)

            #if st.session_state.sorted_df_grp is not None:
            #    st.write('정렬된 상위 ' + str(len(st.session_state.sorted_df_grp)) + '광고그룹')
            #    st.write(st.session_state.sorted_df_grp)
            #if st.session_state.grp_ranking_chain_result is not None:
            #    st.write(st.session_state.grp_ranking_chain_result)
            #if st.session_state.grp_statements:
            #    for statement in st.session_state.grp_statements:
            #        st.write(statement)

        with kwrd_ranking:
            st.header("키워드별 성과 분석")
            st.write("성과 상위 키워드를 분석합니다.")

            with st.spinner('키워드별 데이터...'):
                result = {}
                for index, row in filtered_group.iterrows():
                    key = (row['소재명/키워드'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ch_ad_week = pd.DataFrame(result).T
                if ch_ad_week.index.nlevels == 2:
                    ch_ad_week.index.names = ['소재명/키워드', group_period]
                #ch_ad_week.index.names = ['소재명/키워드', group_period]
                
                cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in filtered_ga_group.iterrows():
                    key = (row['소재명/키워드'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ga_ch_ad_week = pd.DataFrame(result_ga).T

                if ga_ch_ad_week.index.nlevels == 2:
                    ga_ch_ad_week.index.names = ['소재명/키워드', group_period]
                else:
                    st.write("※※※ 업로드하신 데이터에 소재명/키워드 정보가 없습니다. 다른 캠페인을 선택해주세요. ※※※")
                
                ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                ga_cal_ch_ad_week['CPA'] = (cal_ch_ad_week['총비용'] / ga_cal_ch_ad_week['전환수'])
                ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)
                
                ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                df_combined.reset_index(inplace=True)
                df_combined[['소재명/키워드', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                columns = ['소재명/키워드', group_period] + [col for col in df_combined.columns if (col != '소재명/키워드') and (col != group_period)]
                df_cleaned = df_combined.dropna(subset=['소재명/키워드'])
                df_combined_re = df_cleaned[columns]

                
                result = {}
                for index, row in internal_ch_df.iterrows():
                    key = (row['매체'],row['캠페인'],row['광고그룹'], row['소재명/키워드'],row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ch_ad_week = pd.DataFrame(result).T
                i_ch_ad_week.index.names = ['매체','캠페인','광고그룹','소재명/키워드', group_period]
                
                i_cal_ch_ad_week = report_table(i_ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in ga_df.iterrows():
                    key = (row['매체'],row['캠페인'],row['광고그룹'], row['소재명/키워드'],row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ga_ch_ad_week = pd.DataFrame(result_ga).T
                i_ga_ch_ad_week.index.names = ['매체','캠페인','광고그룹','소재명/키워드', group_period]
                
                i_ga_cal_ch_ad_week = ga_report_table(i_ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                i_ga_cal_ch_ad_week['CPA'] = (i_cal_ch_ad_week['총비용'] / i_ga_cal_ch_ad_week['전환수'])
                i_ga_cal_ch_ad_week['CPA'] = pd.to_numeric(i_ga_cal_ch_ad_week['CPA'], errors='coerce')
                i_ga_cal_ch_ad_week['CPA'] = i_ga_cal_ch_ad_week['CPA'].round(0)
                
                i_ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in i_ga_cal_ch_ad_week.columns]

                i_df_combined = pd.concat([i_cal_ch_ad_week, i_ga_cal_ch_ad_week], axis=1)
                i_df_combined.reset_index(inplace=True)
                i_df_combined[['매체','캠페인','광고그룹','소재명/키워드', group_period]] = pd.DataFrame(i_df_combined['index'].tolist(), index=i_df_combined.index)
                i_df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                i_columns = ['매체','캠페인','광고그룹','소재명/키워드', group_period] + [col for col in i_df_combined.columns if  (col != '소재명/키워드') and (col != '매체') and (col != '캠페인') and (col != '광고그룹') and (col != group_period)]
                i_df_cleaned = i_df_combined.dropna(subset=['소재명/키워드'])
                i_df_combined_re = i_df_combined[i_columns]
                
            now_kwrd_da_week = df_combined_re[df_combined_re[group_period] == now_media]
            de_now_kwrd_da_week = i_df_combined_re[i_df_combined_re[group_period] == now_media]
                        
            sort_columns = st.session_state.selected_metric_cmp

            for mtrc in sort_columns:
                st.subheader(f'성과 상위 소재명/키워드 by {mtrc}')
                sorted_da_df = now_kwrd_da_week.sort_values(by=mtrc, ascending=sort_orders[mtrc]).head(5)
                st.dataframe(sorted_da_df[['소재명/키워드', mtrc]])
                filter_list = list(sorted_da_df['소재명/키워드'])
                # 선택된 키워드에 대한 데이터 필터링
                filtered_data = de_now_kwrd_da_week[de_now_kwrd_da_week['소재명/키워드'].isin(filter_list)]
                st.write(filtered_data)

                kwrd_description = "keywords performance results by " + str(mtrc) + " :\n\n"
                kwrd_description += filtered_data.to_string()


                kwrd_prompt = ChatPromptTemplate.from_template(
                        """
                        너는 퍼포먼스 마케팅 성과 분석가야.
                        다음은 {metric}에 따른 성과가 좋은 키워드에 대한 데이터야.
                        \n\n{kwrd_perf}

                        {kwrd_list}를 대상으로 {kwrd_perf}를 분석해서
                        가장 {metric}이 좋은 매체, 캠페인, 광고그룹, 그것의 {metric} 성과를 출력해.

                        한 개의 키워드마다 아래 형태로 출력해줘.
                        -----------
                        키워드
                        ● 매체 : 이름
                        ● 캠페인 : 이름
                        ● 광고그룹 : 이름
                        ● {metric} : 수치

                        각 매체별로 한글로 100자 정도로 표현해줘.
                        제목은 만들지마.
                        출력할 때, 마크다운 만들지마.
                        수치 표현할 때는 천 단위에서 쉼표 넣어줘.

                    """
                    )

                kwrd_chain = kwrd_prompt | media_llm | StrOutputParser()
                with st.status("키워드별 분석...") as status: 
                    descript_kwrd = kwrd_chain.invoke(
                        {"kwrd_list":filter_list,"metric":mtrc,"kwrd_perf":kwrd_description},
                    )
                    
                st.markdown(descript_kwrd)

        with history:
            with st.spinner('운영 히스토리 데이터 분석 중...'):
                st.write(history_df)

            last_period_data = history_df[history_df[group_period] == pre_media]
            current_period_data = history_df[history_df[group_period] == now_media]

            history_prompt = ChatPromptTemplate.from_template(
                        """
                        너는 퍼포먼스 마케팅 성과 분석가야.
                        주어진 운영 히스토리로 인해 성과에 확인해야 하는 것이 무엇인지 안내해줘.

                        다음은 운영 히스토리 데이터야.
                        \n\n{history}
                        
                        그리고 매체에 대한 정보가 없으면 확인할 특별 사항이 없다고 해줘.
                        매체 정보가 있는 경우 확인해야 할 가능성이 높아져.
                        매체를 언급하면서, 유입 성과와 전환 성과 관점에서 안내해줘.

                        한글로 50자 정도로 표현해줘.
                        존댓말을 써야 해.
                    """
                )
            history_chain = history_prompt | strict_llm | StrOutputParser()

            # 지난 기간 데이터를 출력합니다.
            st.subheader('지난 기간')
            for index, row in last_period_data.iterrows():
                st.write(f"- {row['운영 히스토리']}")
            last_history_description = "history of last period:\n\n"
            last_history_description += last_period_data.to_string()
            descript_last_his = history_chain.invoke(
                        {"history":last_history_description,},
                    )
            st.write(descript_last_his)

            # 이번 기간 데이터를 출력합니다.
            st.subheader('이번 기간')
            for index, row in current_period_data.iterrows():
                st.write(f"- {row['운영 히스토리']}")
            current_history_description = "history of current period:\n\n"
            current_history_description += current_period_data.to_string()
            descript_current_his = history_chain.invoke(
                        {"history":current_history_description,},
                    )
            st.write(descript_current_his)

        with preview:
            st.write('coming soon')


    #커머스
    else:
        overview, ch_ranking, cmp_ranking, grp_ranking, kwrd_ranking, history, preview = st.tabs(["오버뷰","매체별 성과","매체 선택 캠페인 분석", "캠페인 선택 그룹 분석", "성과 상위 소재(키워드) 분석", '운영 히스토리',  '프리뷰'])
        internal_ch_df['일자'] = internal_ch_df['일자'].astype(str)

        with overview:
            if st.session_state.overview_result is None:
                st.subheader('오버뷰')
                with st.spinner('데이터 분석 중...'):
                    target_list_media = list_inflow + list_trans_media

                    result = {}
                    for index, row in internal_ch_df.iterrows():
                        category = row[group_period]
                        
                        if category not in result:
                            result[category] = {col: 0 for col in target_list_media}
                        
                        for col in target_list_media:
                            result[category][col] += row[col]
                    ad_week = pd.DataFrame(result).T
                    ad_week.index.name = group_period

                    cal_ad_week = report_table(ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)
                    st.session_state.overview_ad_df_result = cal_ad_week
                    cal_ad_week.loc['변화량'] = cal_ad_week.diff().iloc[1]
                    cal_ad_week.loc['증감율'] = round(((cal_ad_week.loc[now_media] - cal_ad_week.loc[pre_media]) / cal_ad_week.loc[pre_media]) * 100, 2)

                    result_ga = {}
                    for index, row in ga_df.iterrows():
                        category = row[group_period]
                        
                        if category not in result_ga:
                            result_ga[category] = {col: 0 for col in list_trans_ga}
                        
                        for col in list_trans_ga:
                            result_ga[category][col] += row[col]
                    ga_ad_week = pd.DataFrame(result_ga).T
                    ga_ad_week.index.name = group_period

                    ga_cal_ad_week = ga_report_table(ga_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)

                    ga_cal_ad_week['CPA'] = (cal_ad_week['총비용'] / ga_cal_ad_week['전환수'])
                    ga_cal_ad_week['CPA'] = pd.to_numeric(ga_cal_ad_week['CPA'], errors='coerce')
                    ga_cal_ad_week['CPA'] = ga_cal_ad_week['CPA'].round(0)

                    ga_cal_ad_week['ROAS'] = (ga_cal_ad_week['구매액'] / cal_ad_week['총비용']) * 100
                    ga_cal_ad_week['ROAS'] = pd.to_numeric(ga_cal_ad_week['ROAS'], errors='coerce')
                    ga_cal_ad_week['ROAS'] = ga_cal_ad_week['ROAS'].round(0)
                    ga_cal_ad_week['전환율'] = (ga_cal_ad_week['구매'] / cal_ad_week['클릭수']) * 100
                    ga_cal_ad_week['전환율'] = pd.to_numeric(ga_cal_ad_week['전환율'], errors='coerce')
                    ga_cal_ad_week['전환율'] = ga_cal_ad_week['전환율'].round(2)

                    ga_cal_ad_week.columns = [f'GA_{col}' for col in ga_cal_ad_week.columns]

                    st.session_state.overview_ga_ad_df_result = ga_cal_ad_week

                    ga_cal_ad_week.loc['변화량'] = ga_cal_ad_week.diff().iloc[1]
                    ga_cal_ad_week.loc['증감율'] = round(((ga_cal_ad_week.loc[now_media] - ga_cal_ad_week.loc[pre_media]) / ga_cal_ad_week.loc[pre_media]) * 100, 2)

                    

                    # 데이터 프레임을 좌우로 붙이기
                    df_combined = pd.concat([cal_ad_week, ga_cal_ad_week], axis=1)
                    st.session_state.overview_result = df_combined
                    st.write(df_combined)

                description = "Periodical change data results:\n\n"
                description += df_combined.to_string()

                previous_period = df_combined.iloc[0]
                current_period = df_combined.iloc[1]
                change_period = df_combined.iloc[2]
                columns = df_combined.columns[1:]

                # Generating the sentences
                sentences = []
                for col in columns:
                    change = "증가" if change_period[col] > 0 else "감소"
                    sentence = f"{col}은 지난 기간 대비 {abs(change_period[col]):,.2f} {change}하였습니다. ({previous_period[col]:,.2f} -> {current_period[col]:,.2f})"
                    sentences.append(sentence)

            
                month_compare_prompt = ChatPromptTemplate.from_template(
                    """
                    너는 퍼포먼스 마케팅 성과 분석가야.
                    다음 주차에 따른 성과 자료를 기반으로 유입 성과와 전환 성과를 분석해야해.
                    \n\n{description}
                    \n\n{sentences}

                    노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                    회원가입, DB전환, 가망, 전환수, CPA는 매체 전환에 대한 성과야.
                    GA_회원가입, GA_db전환, GA_카톡btn, GA_전화btn, GA_총합계, GA_CPA는 GA 전환에 대한 성과야.

                    첫 행은 비교하기 위한 바로 직전 주 성과이고, 두번째 행은 이번 주차의 성과야.

                    유입 성과는 CTR과 CPC가 얼마나 변하였고, 그에 대한 근거로 노출수와 클릭수, 비용이 어떻게 변화했기에 CTR과 CPC가 그러한 변화를 가지게 되었는지 분석해야해.
                    전환 성과는 전환수가 얼마나 변하였고, CPA가 얼마나 변하였는지를 파악하고, 그에 대한 근거로 노출수, 클릭수, 비용, 회원가입, DB전환, 가망에서의 변화를 분석해야해.
                    매체 전환과 GA 전환을 구분해서 설명해야해.

                    숫자를 사용할 때는 지난 기간의 절대값과 이번 기간의 절대값을 모두 표시해줘.
                    증감율에서 숫자를 인용할 때는 퍼센테이지를 붙여서 설명해야해.
                    1% 이상의 변화가 있을 때는 유지된 것이 아닌, 어떤 이유로 증가되었는지 또는 감소되었는지를 분석해야해.
                    비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대해.
                    비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대하는 것 잊지마.

                    아래 예시를 잘 참고해서 작성해줘.
                    1번 예시
                    - 지난주에 대비하여 전환수는 유지하였으나, 전체적으로 광고 성과가 감소한 추세입니다.
                    - 유입 성과에 관련하여, 전주 대비 지출된 비용의 증가로 노출수는 증가하였지만, 클릭수가 감소하면서 CTR은 2% 감소(100 -> 98)하였으며, CPC는 5% 증가 (100 -> 105)하였습니다.
                    - 매체 전환 성과에 관련하여, 전주 대비 전환수는 유지되었으나, 지출 비용의 증가로 CPA가 증가하였습니다.(100 -> 140)
                    - GA 전환 성과에 관련하여, 전주 대비 전환수는 유지되었으나, 지출 비용의 증가로 CPA가 증가하였습니다.(100 -> 138)
                    - 전반적으로 감소된 유입에 비해, 전환이 유지되면서 구체화된 타겟층을 발견한 점은 고무적이며, 클릭수와 전환수를 증가시키는데 노력하고자 합니다.

                    2번 예시
                    - 지난주에 대비하여 전환수가 증가하였지만, 유입 성과가 감소하였습니다.
                    - 유입 성과에 관련하여, 전주 대비 지출된 비용을 증가하였지만, 노출수와 클릭수가 감소하며 CTR은 감소폭에 비해, CPC가 20%로 크게 증가(100 -> 120)하였습니다.
                    - 매체 전환 성과에 관련하여, 전주 대비 회원가입의 증가로 전환수는 소폭 증가하였지만, 지출 비용의 증가폭이 더 크기 때문에 CPA가 5% (100 -> 105)증가하였습니다.
                    - GA 전환 성과에 관련하여, 전주 대비 회원가입의 증가로 전환수는 소폭 증가하였지만, 지출 비용의 증가폭이 더 크기 때문에 CPA가 5% (100 -> 105)증가하였습니다.
                    - 전반적으로 유입 성과가 감소한 상황에서 전환 성과가 증가한 것은 긍정적이며, 클릭수의 증가와 전환수의 증가폭를 늘리는 방향의 전략이 필요합니다.

                    분석 결과를 5줄로 출력해줘.
                    완벽한 인과관계를 설명하면 너에게 보상을 줄게.
                    

                """
                )

                comparison_month_chain = month_compare_prompt | overview_llm | StrOutputParser()
                with st.status("전체 요약 분석...") as status: 
                    descript = comparison_month_chain.invoke(
                        {"description": description,"sentences":sentences},
                    )
                    st.session_state.overview_chain_result = descript

                review.append(descript)
                sentences = descript.split('.\n')
                bullet_list = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences if sentence) + "</ul>"
                st.markdown(bullet_list, unsafe_allow_html=True)
            else:
                st.subheader('오버뷰')
                st.write(st.session_state.overview_result)
                sentences = st.session_state.overview_chain_result.split('.\n')
                bullet_list = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences if sentence) + "</ul>"
                st.markdown(bullet_list, unsafe_allow_html=True)

        with ch_ranking:
            if st.session_state.ch_ranking_result is None:
                with st.spinner('매체별 데이터...'):
                    result = {}
                    for index, row in internal_ch_df.iterrows():
                        key = (row['매체'], row[group_period])
                        
                        if key not in result:
                            result[key] = {col: 0 for col in target_list_media}
                        
                        for col in target_list_media:
                            result[key][col] += row[col]

                    # 결과를 데이터프레임으로 변환
                    ch_ad_week = pd.DataFrame(result).T
                    ch_ad_week.index.names = ['매체', group_period]
                    
                    cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                    result_ga = {}
                    for index, row in ga_df.iterrows():
                        key = (row['매체'], row[group_period])
                        
                        if key not in result_ga:
                            result_ga[key] = {col: 0 for col in list_trans_ga}
                        
                        for col in list_trans_ga:
                            result_ga[key][col] += row[col]

                    # 결과를 데이터프레임으로 변환
                    ga_ch_ad_week = pd.DataFrame(result_ga).T
                    ga_ch_ad_week.index.names = ['매체', group_period]
                    
                    ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)

                    ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['전환수'].apply(lambda x: cal_ch_ad_week['총비용'][ga_cal_ch_ad_week.index[ga_cal_ch_ad_week['전환수'] == x][0]] / x if x != 0 else 0)
                    ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                    ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)

                    ga_cal_ch_ad_week['ROAS'] = ga_cal_ch_ad_week['구매액'].apply(lambda x: cal_ch_ad_week['총비용'][ga_cal_ch_ad_week.index[ga_cal_ch_ad_week['구매액'] == x][0]] / x * 100 if x != 0 else 0)
                    #ga_cal_ch_ad_week['ROAS'] = (ga_cal_ch_ad_week['구매액'] / cal_ch_ad_week['총비용']) * 100
                    ga_cal_ch_ad_week['ROAS'] = pd.to_numeric(ga_cal_ch_ad_week['ROAS'], errors='coerce')
                    ga_cal_ch_ad_week['ROAS'] = ga_cal_ch_ad_week['ROAS'].round(0)
                    ga_cal_ch_ad_week['전환율'] = (ga_cal_ch_ad_week['구매'] / cal_ch_ad_week['클릭수']) * 100
                    ga_cal_ch_ad_week['전환율'] = pd.to_numeric(ga_cal_ch_ad_week['전환율'], errors='coerce')
                    ga_cal_ch_ad_week['전환율'] = ga_cal_ch_ad_week['전환율'].round(2)
                    
                    ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                    df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                    df_combined.reset_index(inplace=True)
                    df_combined[['매체', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                    df_combined.drop(columns=['index'], inplace=True)
                    # 특정 열을 앞에 오도록 열 순서 재배치
                    columns = ['매체', group_period] + [col for col in df_combined.columns if (col != '매체') and (col != group_period)]
                    df_combined_re = df_combined[columns]

                    result = {}
                    cal_ad_week = st.session_state.overview_ad_df_result
                    ga_cal_ad_week = st.session_state.overview_ga_ad_df_result

                    sum_df_combined = pd.concat([cal_ad_week, ga_cal_ad_week], axis=1)
                
                    st.subheader('기간별 매체 순위 변화')
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader(pre_media)
                        pre_week = df_combined_re[df_combined_re[group_period] == pre_media]
                        pre_week_desc = pre_week.sort_values(by='전환수', ascending=False)
                        # Step 2: Extract a row using loc
                        row = sum_df_combined.loc[pre_media]  # This extracts the second row (index 1)

                        # Convert the extracted row to a DataFrame
                        row_df = pd.DataFrame([row])

                        # Optionally reset the index to maintain consistency
                        row_df.reset_index(drop=True, inplace=True)

                        # Step 3: Concatenate the DataFrame with the extracted row
                        pre_result = pd.concat([pre_week_desc, row_df], axis=0, ignore_index=True)
                        pre_description = "Previous period performance data results:\n\n"
                        pre_description += pre_week_desc.to_string()
                        st.write(pre_result)
                        result['pre_period'] = pre_result
                    with col2:
                        st.subheader(now_media)
                        now_week = df_combined_re[df_combined_re[group_period] == now_media]
                        now_week_desc = now_week.sort_values(by='전환수', ascending=False)
                        # Step 2: Extract a row using loc
                        row = sum_df_combined.loc[now_media]  # This extracts the second row (index 1)

                        # Convert the extracted row to a DataFrame
                        row_df = pd.DataFrame([row])

                        # Optionally reset the index to maintain consistency
                        row_df.reset_index(drop=True, inplace=True)

                        # Step 3: Concatenate the DataFrame with the extracted row
                        now_result = pd.concat([now_week_desc, row_df], axis=0, ignore_index=True)
                        now_description = "This period performance data results:\n\n"
                        now_description += now_week_desc.to_string()
                        st.write(now_result)
                        result['now_period'] = now_result
                    st.session_state.ch_ranking_result = result
                    percentage_changes = {
                        "매체": pre_result["매체"],
                        "노출수 변화율": calculate_percentage_change(pre_result, now_result, "노출수"),
                        "클릭수 변화율": calculate_percentage_change(pre_result, now_result, "클릭수"),
                        "CTR 변화율": calculate_percentage_change(pre_result, now_result, "CTR"),
                        "CPC 변화율": calculate_percentage_change(pre_result, now_result, "CPC"),
                        "총비용 변화율": calculate_percentage_change(pre_result, now_result, "총비용"),
                        "회원가입 변화율": calculate_percentage_change(pre_result, now_result, "회원가입"),
                        "장바구니 변화율": calculate_percentage_change(pre_result, now_result, "장바구니"),
                        "구매 변화율": calculate_percentage_change(pre_result, now_result, "구매"),
                        "구매액 변화율": calculate_percentage_change(pre_result, now_result, "구매액"),
                        "전환수 변화율": calculate_percentage_change(pre_result, now_result, "전환수"),
                        "CPA 변화율": calculate_percentage_change(pre_result, now_result, "CPA")
                    }

                    df_percentage_changes = pd.DataFrame(percentage_changes)
                    df_per_description = "Periodical change data results by channel:\n\n"
                    df_per_description += df_percentage_changes.to_string()

                    # Calculate new influences
                    influences = {
                        "매체": pre_result["매체"],
                        "노출수 영향도": calculate_new_influence(pre_result, now_result, "노출수"),
                        "클릭수 영향도": calculate_new_influence(pre_result, now_result, "클릭수"),
                        "총비용 영향도": calculate_new_influence(pre_result, now_result, "총비용"),
                        "전환수 영향도": calculate_new_influence(pre_result, now_result, "전환수")
                    }

                    df_influences = pd.DataFrame(influences)

                    # Calculate new impact changes
                    impact_changes = {
                        "매체": df_percentage_changes["매체"],
                        "노출수 영향 변화율": df_influences["노출수 영향도"] * df_percentage_changes["노출수 변화율"],
                        "클릭수 영향 변화율": df_influences["클릭수 영향도"] * df_percentage_changes["클릭수 변화율"],
                        "총비용 영향 변화율": df_influences["총비용 영향도"] * df_percentage_changes["총비용 변화율"],
                        "전환수 영향 변화율": df_influences["전환수 영향도"] * df_percentage_changes["전환수 변화율"]
                    }

                    df_impact_changes = pd.DataFrame(impact_changes)

                    df_impact_description = "Periodical change data results influencing by channel:\n\n"
                    df_impact_description += df_impact_changes.to_string()

                    #매체별 성과 증감 비교
                    dic_ch_ad_week = {}
                    dic_description = {}
                    channels = now_week_desc['매체'].unique()

                    for channel in channels:
                        ch_df = df_combined_re[df_combined_re['매체'] == str(channel)]
                        ch_df.set_index(group_period, inplace=True)
                        ch_df.drop(columns=['매체'], inplace=True)
                        #st.write(ch_df)

                        try:
                            ch_df.loc['변화량'] = ch_df.diff().iloc[1]
                        except:
                            st.write("")
                            #st.write("전 기간 또는 해당 기간 정보가 없습니다.")
                        # 새로운 증감율 행 생성
                        increase_rate = []
                        for col in ch_df.columns:
                            try:
                                rate = calculate_increase_rate(ch_df.loc[now_media, col], ch_df.loc[pre_media, col])
                            except:
                                rate = None
                            increase_rate.append(rate)

                        # 데이터프레임에 증감율 행 추가
                        ch_df.loc['증감율'] = increase_rate
                        #ch_df.loc['증감율'] = round(((ch_df.loc['4월 3주'] - ch_df.loc['4월 2주']) / ch_df.loc['4월 2주']) * 100, 2)

                        ch_description = "Periodical change data results in" + str(channel) + " :\n\n"
                        ch_description += ch_df.to_string()

                        dic_ch_ad_week[str(channel)] = ch_df
                        dic_description[str(channel)] = ch_description


                    compare_prompt = ChatPromptTemplate.from_template(
                            """
                            너는 퍼포먼스 마케팅 성과 분석가야.
                            각 매체의 성과 변화를 요약해야해.
                            다음은 지난주에 비해서 각 매체별 지표가 어떻게 변하였는지 나타내.
                            \n\n{overview_per}
                            
                            노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                            회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                            특정 지표의 증감을 이야기 할 때는 증감율을 인용하고 퍼센테이지를 붙여서 설명해야해.

                            아래 예시를 잘 참고해서 작성해줘.
                            1번 예시
                            - 구글: 대부분의 지표가 감소하였으나, 회원가입(10%)은 증가했습니다.
                            - 네이버: 노출수(2%)와 클릭수(3%), 전환수(1%)가 모두 증가하였으나 CPA는 감소(-5%)했습니다.
                            - 모비온: 회원가입(10%)과 DB전환(15%)이 크게 증가했으나 클릭수(-2%)와 CPA(-7%)가 감소했습니다.
                            - 페이스북: 노출수(8%)와 클릭수(3%)가 증가했으나, 전환수(-5%)는 감소했습니다.
                            - 타불라: 노출수(-35%)는 크게 감소했으나, 전환수(4%)가 증가했습니다.
                            - 카카오모먼트: CTR(9%)이 증가하였지만, CPA(25%)가 더 크게 증가하였습니다.
                            - 당근 비즈니스: 노출수(-5%)가 크게 감소했습니다.
                            - 카카오SA: 지난주와 거의 유사합니다.

                            완벽한 인과관계를 설명하면 너에게 보상을 줄게.
                            각 매체별로 한글로 30자 정도로 표현해줘.

                        """
                        )

                    comparison_chain = compare_prompt | media_llm | StrOutputParser()
                    with st.status("매체별 분석...") as status: 
                        descript_ch = comparison_chain.invoke(
                            {"overview_per":df_per_description},
                        )
                        st.session_state.ch_ranking_chain_result = descript_ch
                        
                    sentences_ch = descript_ch.split('.\n')
                    review.append(descript_ch)
                    bullet_list_ch = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch if sentence) + "</ul>"
                    st.markdown(bullet_list_ch, unsafe_allow_html=True)

                    
                    impact_prompt = ChatPromptTemplate.from_template(
                            """
                            너는 퍼포먼스 마케팅 성과 분석가야.
                            각 매체의 성과 변화가 얼마나 영향을 미쳤는지 요약해야해.
                            다음은 지난주에 비해서 각 매체별 지표가 어떻게 변하였고 그 영향력이 어느 정도였는지 나타내.
                            {overview_im}
                            
                            노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                            회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                            유입 성과는 CTR과 CPC가 얼마나 변하였고, 그에 대한 근거로 노출수와 클릭수, 비용이 어떻게 변화했기에 CTR과 CPC가 그러한 변화를 가지게 되었는지 분석해야해.
                            클릭수가 증가했을 때, 노출수가 클릭수에 비해서 크게 증가하면 CTR이 감소해.
                            클릭수가 증가했을 때, 노출수가 감소하면 CTR이 증가해.
                            비용의 증가는 노출수의 증가와 그로 이한 클릭수의 증가를 기대해.
                            전환 성과는 전환수가 얼마나 변하였고, CPA가 얼마나 변하였는지를 파악하고, 그에 대한 근거로 노출수, 클릭수, 비용, 회원가입, DB전환, 가망에서의 변화를 분석해야해.

                            아래 예시를 잘 참고해서 작성해줘.
                            1번 예시
                            - 네이버와 카카오SA의 비용의 증가가 비교적 컸지만, 기대한 노출수와 클릭수의 증가로 이어지지 않았습니다. 그러나, 구글 성과의 감소에도 불구하고 네이버와 모비온에서의 전환수 증가로 전환 성과가 향상되었습니다.
                            2번 예시
                            - 전환수가 가장 높은 구글의 전체적인 성과 감소로 전체 성과의 감소 우려가 있었으나, 네이버, 모비온, 타불라의 전환 성과가 향상되며 전주와 유사한 성과를 유지할 수 있었습니다.
                            3번 예시
                            - 페이스북과 당근비즈니스, 카카오모먼트는 성과 변화가 크지 않았습니다.

                            완벽한 인과관계를 설명하면 너에게 보상을 줄게.
                            한글로 150자 정도로 표현해줘.

                        """
                        )

                    impact_chain = impact_prompt | influence_llm | StrOutputParser()
                    with st.status("영향력 분석...") as status: 
                        descript_im = impact_chain.invoke(
                            {"overview_im":df_impact_description},
                        )
                        st.session_state.ch_ranking_influence_analysis = descript_im

                    sentences_im = descript_im.split('.\n')
                    bullet_list_im = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_im if sentence) + "</ul>"
                    st.markdown(bullet_list_im, unsafe_allow_html=True)

                
                    st.subheader('매체별 변화량 비교')


                    for channel in channels:
                        st.subheader(channel)
                        st.write(dic_ch_ad_week[channel])

                        ch_compare_prompt = ChatPromptTemplate.from_template(
                            """
                            너는 퍼포먼스 마케팅 성과 분석가야.
                            다음 주차에 따른 성과 자료를 기반으로 유입 성과와 전환 성과를 분석해야해.
                            \n\n{description_ch}

                            노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                            회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                            첫 행은 비교하기 위한 바로 직전 주 성과이고, 두번째 행은 이번 주차의 성과야.

                            유입 성과는 CTR과 CPC가 얼마나 변하였고, 그에 대한 근거로 노출수와 클릭수, 비용이 어떻게 변화했기에 CTR과 CPC가 그러한 변화를 가지게 되었는지 분석해야해.
                            전환 성과는 전환수가 얼마나 변하였고, CPA가 얼마나 변하였는지를 파악하고, 그에 대한 근거로 노출수, 클릭수, 비용, 회원가입, DB전환, 가망에서의 변화를 분석해야해.

                            증감율에서 숫자를 인용할 때는 퍼센테이지를 붙여서 설명해야해.
                            1% 이상의 변화가 있을 때는 유지된 것이 아닌, 어떤 이유로 증가되었는지 또는 감소되었는지를 분석해야해.
                            비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대해.
                            비용의 증가는 노출수, 클릭수, 전환수의 증가를 기대하는 것 잊지마.

                            분석 결과를 2줄로 출력해줘.
                            완벽한 인과관계를 설명하면 너에게 보상을 줄게.

                        """
                        )

                        comparison_ch_chain = ch_compare_prompt | strict_llm | StrOutputParser()
                        with st.status("매체별 분석 중..." + channel) as status: 
                            descript_ch_ad = comparison_ch_chain.invoke(
                                {"description_ch": dic_description[channel]},
                            )
                            st.session_state.ch_ranking_individual_results[channel] = {
                            'dataframe': ch_df,
                            'analysis': descript_ch_ad
                            }  
                        
                        sentences_ch_ad = descript_ch_ad.split('.\n')
                        bullet_list_ch_ad = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch_ad if sentence) + "</ul>"
                        st.markdown(bullet_list_ch_ad, unsafe_allow_html=True)
            else:
                st.subheader('기간별 매체 순위 변화')
                col1, col2 = st.columns(2)
                result = st.session_state.ch_ranking_result
                with col1:
                    st.subheader(pre_media)
                    st.write(result['pre_period'])
                with col2:
                    st.subheader(now_media)
                    st.write(result['now_period'])

                sentences_ch = st.session_state.ch_ranking_chain_result.split('.\n')
                bullet_list_ch = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch if sentence) + "</ul>"
                st.markdown(bullet_list_ch, unsafe_allow_html=True)
                st.subheader('영향력 분석')
                sentences_im = st.session_state.ch_ranking_influence_analysis.split('.\n')
                bullet_list_im = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_im if sentence) + "</ul>"
                st.markdown(bullet_list_im, unsafe_allow_html=True)

                for channel in st.session_state.ch_ranking_individual_results:
                    st.subheader(channel)
                    st.write(st.session_state.ch_ranking_individual_results[channel]['dataframe'])
                    sentences_ch_ad = st.session_state.ch_ranking_individual_results[channel]['analysis'].split('.\n')
                    bullet_list_ch_ad = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences_ch_ad if sentence) + "</ul>"
                    st.markdown(bullet_list_ch_ad, unsafe_allow_html=True)

        with cmp_ranking:
            st.header("캠페인 분석")
            st.write("분석하고자 하는 광고유형을 선택해주세요.")
            selected_ad_type = st.selectbox("광고유형 선택", internal_ch_df["광고유형"].unique())
            st.session_state.selected_ad_type = selected_ad_type

            filtered_by_ad_type = internal_ch_df[internal_ch_df["광고유형"] == selected_ad_type]

            st.write("분석하고자 하는 매체를 선택해주세요.")
            selected_media = st.radio("매체 선택", filtered_by_ad_type["매체"].unique())
            st.session_state.selected_media_cmp = selected_media

            filtered_br = internal_ch_df[internal_ch_df["매체"] == selected_media]
            filtered_ga_br = ga_df[ga_df["매체"] == selected_media]
            with st.spinner('캠페인별 데이터...'):
                result = {}
                for index, row in filtered_br.iterrows():
                    key = (row['캠페인'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ch_ad_week = pd.DataFrame(result).T
                if ch_ad_week.index.nlevels == 2:
                    ch_ad_week.index.names = ['캠페인', group_period]
                #ch_ad_week.index.names = ['캠페인', group_period]
                
                cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                st.write(cal_ch_ad_week)

                result_ga = {}
                for index, row in filtered_ga_br.iterrows():
                    key = (row['캠페인'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ga_ch_ad_week = pd.DataFrame(result_ga).T
                #st.write(ga_ch_ad_week,type(ga_ch_ad_week.index.nlevels),ga_ch_ad_week.index)
                if ga_ch_ad_week.index.nlevels == 2:
                    ga_ch_ad_week.index.names = ['캠페인', group_period]
                    #st.write(ga_ch_ad_week)
                else:
                    st.write("※※※ 업로드하신 데이터에 캠페인 정보가 없습니다. 다른 매체를 선택해주세요. ※※※")
                
                ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                st.write(ga_cal_ch_ad_week)
                ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['전환수'].apply(lambda x: cal_ch_ad_week['총비용'][ga_cal_ch_ad_week.index[ga_cal_ch_ad_week['전환수'] == x][0]] / x if x != 0 else 0)
                #ga_cal_ch_ad_week['CPA'] = (cal_ch_ad_week['총비용'] / ga_cal_ch_ad_week['전환수'])
                ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)

                ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['구매액'].apply(lambda x: cal_ch_ad_week['총비용'][ga_cal_ch_ad_week.index[ga_cal_ch_ad_week['구매액'] == x][0]] / x * 100 if x != 0 else 0)
                #ga_cal_ch_ad_week['ROAS'] = (ga_cal_ch_ad_week['구매액'] / cal_ch_ad_week['총비용']) * 100
                ga_cal_ch_ad_week['ROAS'] = pd.to_numeric(ga_cal_ch_ad_week['ROAS'], errors='coerce')
                ga_cal_ch_ad_week['ROAS'] = ga_cal_ch_ad_week['ROAS'].round(0)
                ga_cal_ch_ad_week['전환율'] = (ga_cal_ch_ad_week['구매'] / cal_ch_ad_week['클릭수']) * 100
                ga_cal_ch_ad_week['전환율'] = pd.to_numeric(ga_cal_ch_ad_week['전환율'], errors='coerce')
                ga_cal_ch_ad_week['전환율'] = ga_cal_ch_ad_week['전환율'].round(2)
                
                ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                df_combined.reset_index(inplace=True)
                df_combined[['캠페인', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                columns = ['캠페인', group_period] + [col for col in df_combined.columns if (col != '캠페인') and (col != group_period)]
                df_combined_re = df_combined[columns]

                
                result = {}
                for index, row in internal_ch_df.iterrows():
                    key = (row['캠페인'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ch_ad_week = pd.DataFrame(result).T
                i_ch_ad_week.index.names = ['캠페인', group_period]
                
                i_cal_ch_ad_week = report_table(i_ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in ga_df.iterrows():
                    key = (row['캠페인'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ga_ch_ad_week = pd.DataFrame(result_ga).T
                i_ga_ch_ad_week.index.names = ['캠페인', group_period]
                
                i_ga_cal_ch_ad_week = ga_report_table(i_ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                i_ga_cal_ch_ad_week['CPA'] = i_ga_cal_ch_ad_week['전환수'].apply(lambda x: i_cal_ch_ad_week['총비용'][i_ga_cal_ch_ad_week.index[i_ga_cal_ch_ad_week['전환수'] == x][0]] / x if x != 0 else 0)
                #i_ga_cal_ch_ad_week['CPA'] = (i_cal_ch_ad_week['총비용'] / i_ga_cal_ch_ad_week['전환수'])
                i_ga_cal_ch_ad_week['CPA'] = pd.to_numeric(i_ga_cal_ch_ad_week['CPA'], errors='coerce')
                i_ga_cal_ch_ad_week['CPA'] = i_ga_cal_ch_ad_week['CPA'].round(0)

                i_ga_cal_ch_ad_week['ROAS'] = i_ga_cal_ch_ad_week['구매액'].apply(lambda x: i_cal_ch_ad_week['총비용'][i_ga_cal_ch_ad_week.index[i_ga_cal_ch_ad_week['구매액'] == x][0]] / x * 100 if x != 0 else 0)
                i_ga_cal_ch_ad_week['ROAS'] = (i_ga_cal_ch_ad_week['구매액'] / i_cal_ch_ad_week['총비용']) * 100
                i_ga_cal_ch_ad_week['ROAS'] = pd.to_numeric(i_ga_cal_ch_ad_week['ROAS'], errors='coerce')
                i_ga_cal_ch_ad_week['ROAS'] = i_ga_cal_ch_ad_week['ROAS'].round(0)
                i_ga_cal_ch_ad_week['전환율'] = (i_ga_cal_ch_ad_week['구매'] / i_cal_ch_ad_week['클릭수']) * 100
                i_ga_cal_ch_ad_week['전환율'] = pd.to_numeric(i_ga_cal_ch_ad_week['전환율'], errors='coerce')
                i_ga_cal_ch_ad_week['전환율'] = i_ga_cal_ch_ad_week['전환율'].round(2)
                
                i_ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in i_ga_cal_ch_ad_week.columns]

                i_df_combined = pd.concat([i_cal_ch_ad_week, i_ga_cal_ch_ad_week], axis=1)
                i_df_combined.reset_index(inplace=True)
                i_df_combined[['캠페인', group_period]] = pd.DataFrame(i_df_combined['index'].tolist(), index=i_df_combined.index)
                i_df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                i_columns = ['캠페인', group_period] + [col for col in i_df_combined.columns if (col != '캠페인') and (col != group_period)]
                i_df_combined_re = i_df_combined[i_columns]
                
            now_ch_cmp_week = df_combined_re[df_combined_re[group_period] == now_media]
            i_now_ch_cmp_week = i_df_combined_re[i_df_combined_re[group_period] == now_media]

            
            # 폼 사용
            with st.form(key='sort_form'):
                sort_columns = st.multiselect('가장 먼저 정렬하고 싶은 순서대로 정렬할 기준을 선택하세요 (여러 개 선택 가능):', metric)
                
                # 폼 제출 버튼
                submit_button = st.form_submit_button(label='정렬 적용')

            # 폼이 제출된 경우 정렬 수행
            if submit_button:
                st.session_state.selected_metric_cmp = sort_columns
                ascending_orders = [sort_orders[col] for col in sort_columns]
                
                # 데이터 프레임 정렬
                num_data = len(now_ch_cmp_week)
                if num_data >= 10:
                    sorted_df = now_ch_cmp_week.sort_values(by=sort_columns, ascending=ascending_orders).head(10)
                else:
                    sorted_df = now_ch_cmp_week.sort_values(by=sort_columns, ascending=ascending_orders).head(num_data)

                st.session_state.sorted_df_cmp = sorted_df
                top_num = len(sorted_df)
                statements = generate_statements(sorted_df, i_now_ch_cmp_week, sort_columns, top_num)
                # 정렬된 데이터 프레임 출력
                st.session_state.cmp_statements = statements
                st.write('정렬된 상위 ' + str(top_num) + '개 캠페인')
                st.write(sorted_df)

                metric_str = 'and'.join(str(x) for x in sort_columns)
                cmp_description = "Top " +str(top_num) + " br sorted by " + metric_str + ":\n\n"
                cmp_description += sorted_df.to_string()

                # 값 컬럼을 기준으로 내림차순 정렬 후 상위 10개의 합 계산
                top_10_cost_sum = sorted_df['총비용'].sum()
                total_cost_sum = i_now_ch_cmp_week['총비용'].sum()
                ratio_cost = round((top_10_cost_sum / total_cost_sum) * 100, 2)

                top_10_cv_sum = sorted_df['전환수'].sum()
                total_cv_sum = i_now_ch_cmp_week['전환수'].sum()
                ratio_cv = round((top_10_cv_sum / total_cv_sum) * 100, 2)

                cost_statement = "정렬된 상위 " +str(top_num) + " 개의 총비용("+"{:,}".format(top_10_cost_sum)+")"+ "은 당 기간 전체 집행 비용("+"{:,}".format(total_cost_sum)+")의 "+str(ratio_cost)+"% 입니다."
                cv_statement = "정렬된 상위 " +str(top_num) + " 개의 전환수("+"{:,}".format(top_10_cv_sum)+")는 당 기간 전체 전환수("+"{:,}".format(total_cv_sum)+")의 "+str(ratio_cv)+"% 입니다."

                st.session_state.cmp_statements.insert(0,cv_statement)
                st.session_state.cmp_statements.insert(0,cost_statement)

                #st.write(cost_statement)
                #st.write(cv_statement)
                for statement in statements:
                    st.write(statement)

                campaign_prompt = ChatPromptTemplate.from_template(
                        """
                        너는 퍼포먼스 마케팅 성과 분석가야.
                        각 캠페인의 성과를 요약해야해.
                        다음은 선택한 정렬 기준에 따르
                        상위 {n}개의 캠페인에 대한 성과 데이터야.
                        \n\n{campaign_per}
                        
                        노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                        회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                        각 캠페인에 대한 성과를 분석해서 알려줘.
                    """
                    )

                campaign_chain = campaign_prompt | media_llm | StrOutputParser()
                with st.status("캠페인별 분석...") as status: 
                    descript_cmp = campaign_chain.invoke(
                        {"n":top_num,
                        "campaign_per":cmp_description},
                    )
                st.session_state.cmp_ranking_chain_result = descript_cmp    
                st.write(descript_cmp)

            else:
                st.write('정렬 기준 지표를 선택하세요.')
                if st.session_state.sorted_df_cmp is not None:
                    st.write('정렬된 상위 ' + str(len(st.session_state.sorted_df_cmp)) + '개 캠페인')
                    st.write(st.session_state.sorted_df_cmp)
                if st.session_state.cmp_statements:
                    for statement in st.session_state.cmp_statements:
                        st.write(statement)
                if st.session_state.cmp_ranking_chain_result is not None:
                    st.write(st.session_state.cmp_ranking_chain_result)

        with grp_ranking:
            st.header("그룹 분석")
            st.write("분석하고자 하는 매체와 캠페인을 선택해주세요.")
            selected_media = st.session_state.selected_media_cmp
            #selected_media = st.radio("매체 선택", internal_ch_df["매체"].unique(), key='tab3_media')
            selected_campaign = st.selectbox("캠페인 선택", internal_ch_df[internal_ch_df["매체"] == selected_media]["캠페인"].unique(), key='tab3_campaign')
            st.session_state.selected_campaign_cmp = selected_campaign
            filtered_group = internal_ch_df[(internal_ch_df["매체"] == selected_media) & (internal_ch_df["캠페인"] == selected_campaign)]
            filtered_ga_group = ga_df[(ga_df["매체"] == selected_media) & (ga_df["캠페인"] == selected_campaign)]

            with st.spinner('광고그룹별 데이터...'):
                result = {}
                for index, row in filtered_group.iterrows():
                    key = (row['광고그룹'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ch_ad_week = pd.DataFrame(result).T
                if ch_ad_week.index.nlevels == 2:
                    ch_ad_week.index.names = ['광고그룹', group_period]
                #ch_ad_week.index.names = ['광고그룹', group_period]
                
                cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in filtered_ga_group.iterrows():
                    key = (row['광고그룹'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ga_ch_ad_week = pd.DataFrame(result_ga).T

                if ga_ch_ad_week.index.nlevels == 2:
                    ga_ch_ad_week.index.names = ['광고그룹', group_period]
                else:
                    st.write("※※※ 업로드하신 데이터에 광고그룹 정보가 없습니다. 다른 캠페인을 선택해주세요. ※※※")
                
                ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                ga_cal_ch_ad_week['CPA'] = (cal_ch_ad_week['총비용'] / ga_cal_ch_ad_week['전환수'])
                ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)

                ga_cal_ad_week['ROAS'] = (ga_cal_ad_week['구매액'] / cal_ad_week['총비용']) * 100
                ga_cal_ad_week['ROAS'] = pd.to_numeric(ga_cal_ad_week['ROAS'], errors='coerce')
                ga_cal_ad_week['ROAS'] = ga_cal_ad_week['ROAS'].round(0)
                ga_cal_ad_week['전환율'] = (ga_cal_ad_week['구매'] / cal_ad_week['클릭수']) * 100
                ga_cal_ad_week['전환율'] = pd.to_numeric(ga_cal_ad_week['전환율'], errors='coerce')
                ga_cal_ad_week['전환율'] = ga_cal_ad_week['전환율'].round(2)
                
                ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                df_combined.reset_index(inplace=True)
                df_combined[['광고그룹', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                columns = ['광고그룹', group_period] + [col for col in df_combined.columns if (col != '광고그룹') and (col != group_period)]
                df_combined_re = df_combined[columns]

                
                result = {}
                for index, row in internal_ch_df.iterrows():
                    key = (row['광고그룹'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ch_ad_week = pd.DataFrame(result).T
                i_ch_ad_week.index.names = ['광고그룹', group_period]
                
                i_cal_ch_ad_week = report_table(i_ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in ga_df.iterrows():
                    key = (row['광고그룹'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ga_ch_ad_week = pd.DataFrame(result_ga).T
                i_ga_ch_ad_week.index.names = ['광고그룹', group_period]
                
                i_ga_cal_ch_ad_week = ga_report_table(i_ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                i_ga_cal_ch_ad_week['CPA'] = (i_cal_ch_ad_week['총비용'] / i_ga_cal_ch_ad_week['전환수'])
                i_ga_cal_ch_ad_week['CPA'] = pd.to_numeric(i_ga_cal_ch_ad_week['CPA'], errors='coerce')
                i_ga_cal_ch_ad_week['CPA'] = i_ga_cal_ch_ad_week['CPA'].round(0)

                i_ga_cal_ch_ad_week['ROAS'] = (i_ga_cal_ch_ad_week['구매액'] / cal_ad_week['총비용']) * 100
                i_ga_cal_ch_ad_week['ROAS'] = pd.to_numeric(i_ga_cal_ch_ad_week['ROAS'], errors='coerce')
                i_ga_cal_ch_ad_week['ROAS'] = i_ga_cal_ch_ad_week['ROAS'].round(0)
                i_ga_cal_ch_ad_week['전환율'] = (i_ga_cal_ch_ad_week['구매'] / cal_ad_week['클릭수']) * 100
                i_ga_cal_ch_ad_week['전환율'] = pd.to_numeric(i_ga_cal_ch_ad_week['전환율'], errors='coerce')
                i_ga_cal_ch_ad_week['전환율'] = i_ga_cal_ch_ad_week['전환율'].round(2)
                
                i_ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in i_ga_cal_ch_ad_week.columns]

                i_df_combined = pd.concat([i_cal_ch_ad_week, i_ga_cal_ch_ad_week], axis=1)
                i_df_combined.reset_index(inplace=True)
                i_df_combined[['광고그룹', group_period]] = pd.DataFrame(i_df_combined['index'].tolist(), index=i_df_combined.index)
                i_df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                i_columns = ['광고그룹', group_period] + [col for col in i_df_combined.columns if (col != '광고그룹') and (col != group_period)]
                i_df_combined_re = i_df_combined[i_columns]
                
            now_ch_group_week = df_combined_re[df_combined_re[group_period] == now_media]
            i_now_ch_group_week = i_df_combined_re[i_df_combined_re[group_period] == now_media]


            sort_columns = st.session_state.selected_metric_cmp  # 선택한 지표 상태 불러오기
            ascending_orders = [sort_orders[col] for col in sort_columns]
            num_data = len(now_ch_group_week)
            sorted_group_df = now_ch_group_week.sort_values(by=sort_columns, ascending=ascending_orders).head(10) if num_data >= 10 else now_ch_group_week.sort_values(by=sort_columns, ascending=ascending_orders).head(num_data)
            st.session_state.sorted_df_grp = sorted_group_df
            top_num = len(sorted_group_df)
            statements = generate_statements(sorted_group_df, i_now_ch_group_week, sort_columns, top_num)
            st.session_state.grp_statements = statements

            st.write('정렬된 상위 ' + str(top_num) + '광고그룹')
            st.write(sorted_group_df)

            metric_str = 'and'.join(str(x) for x in sort_columns)
            group_description = "Top " + str(top_num) + " groups sorted by " + metric_str + ":\n\n" + sorted_group_df.to_string()

            top_10_cost_sum = sorted_group_df['총비용'].sum()
            total_cost_sum = i_now_ch_group_week['총비용'].sum()
            ratio_cost = round((top_10_cost_sum / total_cost_sum) * 100, 2)

            top_10_cv_sum = sorted_group_df['전환수'].sum()
            total_cv_sum = i_now_ch_group_week['전환수'].sum()
            ratio_cv = round((top_10_cv_sum / total_cv_sum) * 100, 2)

            cost_statement = "정렬된 상위 " + str(top_num) + "개의 총비용(" + "{:,}".format(top_10_cost_sum) + ")은 당 기간 전체 집행 비용(" + "{:,}".format(total_cost_sum) + ")의 " + str(ratio_cost) + "% 입니다."
            cv_statement = "정렬된 상위 " + str(top_num) + "개의 전환수(" + "{:,}".format(top_10_cv_sum) + ")는 당 기간 전체 전환수(" + "{:,}".format(total_cv_sum) + ")의 " + str(ratio_cv) + "% 입니다."

            
            st.session_state.grp_statements.insert(0,cv_statement)
            st.session_state.grp_statements.insert(0,cost_statement)

            st.write(cost_statement)
            st.write(cv_statement)
            for statement in statements:
                st.write(statement)

            adgroup_prompt = ChatPromptTemplate.from_template(
                """
                너는 퍼포먼스 마케팅 성과 분석가야.
                각 광고그룹의 성과를 요약해야해.
                다음은 선택한 정렬 기준에 따르
                상위 {n}개 광고그룹에 대한 성과 데이터야.
                \n\n{adgroup_per}
                
                노출수, 클릭수, CTR, CPC, 총비용은 유입에 대한 성과야.
                회원가입, DB전환, 가망, 전환수, CPA는 전환에 대한 성과야.

                각 광고그룹에 대한 성과를 분석해서 알려줘.
                """
            )

            adgroup_chain = adgroup_prompt | media_llm | StrOutputParser()
            with st.status("광고그룹별 분석...") as status: 
                descript_group = adgroup_chain.invoke(
                    {"n": top_num, "adgroup_per": group_description},
                )
            st.session_state.grp_ranking_chain_result = descript_group
            st.write(descript_group)

            #if st.session_state.sorted_df_grp is not None:
            #    st.write('정렬된 상위 ' + str(len(st.session_state.sorted_df_grp)) + '광고그룹')
            #    st.write(st.session_state.sorted_df_grp)
            #if st.session_state.grp_ranking_chain_result is not None:
            #    st.write(st.session_state.grp_ranking_chain_result)
            #if st.session_state.grp_statements:
            #    for statement in st.session_state.grp_statements:
            #        st.write(statement)

        with kwrd_ranking:
            st.header("키워드별 성과 분석")
            st.write("성과 상위 키워드를 분석합니다.")

            with st.spinner('키워드별 데이터...'):
                result = {}
                for index, row in filtered_group.iterrows():
                    key = (row['소재명/키워드'], row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ch_ad_week = pd.DataFrame(result).T
                if ch_ad_week.index.nlevels == 2:
                    ch_ad_week.index.names = ['소재명/키워드', group_period]
                #ch_ad_week.index.names = ['소재명/키워드', group_period]
                
                cal_ch_ad_week = report_table(ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in filtered_ga_group.iterrows():
                    key = (row['소재명/키워드'], row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                ga_ch_ad_week = pd.DataFrame(result_ga).T

                if ga_ch_ad_week.index.nlevels == 2:
                    ga_ch_ad_week.index.names = ['소재명/키워드', group_period]
                else:
                    st.write("※※※ 업로드하신 데이터에 소재명/키워드 정보가 없습니다. 다른 캠페인을 선택해주세요. ※※※")
                
                ga_cal_ch_ad_week = ga_report_table(ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                ga_cal_ch_ad_week['CPA'] = (cal_ch_ad_week['총비용'] / ga_cal_ch_ad_week['전환수'])
                ga_cal_ch_ad_week['CPA'] = pd.to_numeric(ga_cal_ch_ad_week['CPA'], errors='coerce')
                ga_cal_ch_ad_week['CPA'] = ga_cal_ch_ad_week['CPA'].round(0)

                ga_cal_ad_week['ROAS'] = (ga_cal_ad_week['구매액'] / cal_ad_week['총비용']) * 100
                ga_cal_ad_week['ROAS'] = pd.to_numeric(ga_cal_ad_week['ROAS'], errors='coerce')
                ga_cal_ad_week['ROAS'] = ga_cal_ad_week['ROAS'].round(0)
                ga_cal_ad_week['전환율'] = (ga_cal_ad_week['구매'] / cal_ad_week['클릭수']) * 100
                ga_cal_ad_week['전환율'] = pd.to_numeric(ga_cal_ad_week['전환율'], errors='coerce')
                ga_cal_ad_week['전환율'] = ga_cal_ad_week['전환율'].round(2)
                
                ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in ga_cal_ch_ad_week.columns]

                df_combined = pd.concat([cal_ch_ad_week, ga_cal_ch_ad_week], axis=1)
                df_combined.reset_index(inplace=True)
                df_combined[['소재명/키워드', group_period]] = pd.DataFrame(df_combined['index'].tolist(), index=df_combined.index)
                df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                columns = ['소재명/키워드', group_period] + [col for col in df_combined.columns if (col != '소재명/키워드') and (col != group_period)]
                df_cleaned = df_combined.dropna(subset=['소재명/키워드'])
                df_combined_re = df_cleaned[columns]
                
                result = {}
                for index, row in internal_ch_df.iterrows():
                    key = (row['매체'],row['캠페인'],row['광고그룹'], row['소재명/키워드'],row[group_period])
                    
                    if key not in result:
                        result[key] = {col: 0 for col in target_list_media}
                    
                    for col in target_list_media:
                        result[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ch_ad_week = pd.DataFrame(result).T
                i_ch_ad_week.index.names = ['매체','캠페인','광고그룹','소재명/키워드', group_period]
                
                i_cal_ch_ad_week = report_table(i_ch_ad_week, list_inflow, list_trans_media, selected_trans_media, commerce_or_not)

                result_ga = {}
                for index, row in ga_df.iterrows():
                    key = (row['매체'],row['캠페인'],row['광고그룹'], row['소재명/키워드'],row[group_period])
                    
                    if key not in result_ga:
                        result_ga[key] = {col: 0 for col in list_trans_ga}
                    
                    for col in list_trans_ga:
                        result_ga[key][col] += row[col]

                # 결과를 데이터프레임으로 변환
                i_ga_ch_ad_week = pd.DataFrame(result_ga).T
                i_ga_ch_ad_week.index.names = ['매체','캠페인','광고그룹','소재명/키워드', group_period]
                
                i_ga_cal_ch_ad_week = ga_report_table(i_ga_ch_ad_week, list_trans_ga, selected_trans_ga, commerce_or_not)
                
                i_ga_cal_ch_ad_week['CPA'] = (i_cal_ch_ad_week['총비용'] / i_ga_cal_ch_ad_week['전환수'])
                i_ga_cal_ch_ad_week['CPA'] = pd.to_numeric(i_ga_cal_ch_ad_week['CPA'], errors='coerce')
                i_ga_cal_ch_ad_week['CPA'] = i_ga_cal_ch_ad_week['CPA'].round(0)

                i_ga_cal_ch_ad_week['ROAS'] = (i_ga_cal_ch_ad_week['구매액'] / cal_ad_week['총비용']) * 100
                i_ga_cal_ch_ad_week['ROAS'] = pd.to_numeric(i_ga_cal_ch_ad_week['ROAS'], errors='coerce')
                i_ga_cal_ch_ad_week['ROAS'] = i_ga_cal_ch_ad_week['ROAS'].round(0)
                i_ga_cal_ch_ad_week['전환율'] = (i_ga_cal_ch_ad_week['구매'] / cal_ad_week['클릭수']) * 100
                i_ga_cal_ch_ad_week['전환율'] = pd.to_numeric(i_ga_cal_ch_ad_week['전환율'], errors='coerce')
                i_ga_cal_ch_ad_week['전환율'] = i_ga_cal_ch_ad_week['전환율'].round(2)
                
                i_ga_cal_ch_ad_week.columns = [f'GA_{col}' for col in i_ga_cal_ch_ad_week.columns]

                i_df_combined = pd.concat([i_cal_ch_ad_week, i_ga_cal_ch_ad_week], axis=1)
                i_df_combined.reset_index(inplace=True)
                i_df_combined[['매체','캠페인','광고그룹','소재명/키워드', group_period]] = pd.DataFrame(i_df_combined['index'].tolist(), index=i_df_combined.index)
                i_df_combined.drop(columns=['index'], inplace=True)
                # 특정 열을 앞에 오도록 열 순서 재배치
                i_columns = ['매체','캠페인','광고그룹','소재명/키워드', group_period] + [col for col in i_df_combined.columns if  (col != '소재명/키워드') and (col != '매체') and (col != '캠페인') and (col != '광고그룹') and (col != group_period)]
                i_df_cleaned = i_df_combined.dropna(subset=['소재명/키워드'])
                i_df_combined_re = i_df_combined[i_columns]
                
            now_kwrd_da_week = df_combined_re[df_combined_re[group_period] == now_media]
            de_now_kwrd_da_week = i_df_combined_re[i_df_combined_re[group_period] == now_media]
                        
            sort_columns = st.session_state.selected_metric_cmp

            for mtrc in sort_columns:
                st.subheader(f'성과 상위 소재명/키워드 by {mtrc}')
                sorted_da_df = now_kwrd_da_week.sort_values(by=mtrc, ascending=sort_orders[mtrc]).head(5)
                st.dataframe(sorted_da_df[['소재명/키워드', mtrc]])
                filter_list = list(sorted_da_df['소재명/키워드'])
                # 선택된 키워드에 대한 데이터 필터링
                filtered_data = de_now_kwrd_da_week[de_now_kwrd_da_week['소재명/키워드'].isin(filter_list)]
                st.write(filtered_data)

                kwrd_description = "keywords performance results by " + str(mtrc) + " :\n\n"
                kwrd_description += filtered_data.to_string()


                kwrd_prompt = ChatPromptTemplate.from_template(
                        """
                        너는 퍼포먼스 마케팅 성과 분석가야.
                        다음은 {metric}에 따른 성과가 좋은 키워드에 대한 데이터야.
                        \n\n{kwrd_perf}

                        {kwrd_list}를 대상으로 {kwrd_perf}를 분석해서
                        가장 {metric}이 좋은 매체, 캠페인, 광고그룹, 그것의 {metric} 성과를 출력해.

                        한 개의 키워드마다 아래 형태로 출력해줘.
                        -----------
                        키워드
                        ● 매체 : 이름
                        ● 캠페인 : 이름
                        ● 광고그룹 : 이름
                        ● {metric} : 수치

                        각 매체별로 한글로 100자 정도로 표현해줘.
                        제목은 만들지마.
                        출력할 때, 마크다운 만들지마.
                        수치 표현할 때는 천 단위에서 쉼표 넣어줘.

                    """
                    )

                kwrd_chain = kwrd_prompt | media_llm | StrOutputParser()
                with st.status("키워드별 분석...") as status: 
                    descript_kwrd = kwrd_chain.invoke(
                        {"kwrd_list":filter_list,"metric":mtrc,"kwrd_perf":kwrd_description},
                    )
                    
                st.markdown(descript_kwrd)

        with history:
            with st.spinner('운영 히스토리 데이터 분석 중...'):
                st.write(history_df)

            last_period_data = history_df[history_df[group_period] == pre_media]
            current_period_data = history_df[history_df[group_period] == now_media]

            history_prompt = ChatPromptTemplate.from_template(
                        """
                        너는 퍼포먼스 마케팅 성과 분석가야.
                        주어진 운영 히스토리로 인해 성과에 확인해야 하는 것이 무엇인지 안내해줘.

                        다음은 운영 히스토리 데이터야.
                        \n\n{history}
                        
                        그리고 매체에 대한 정보가 없으면 확인할 특별 사항이 없다고 해줘.
                        매체 정보가 있는 경우 확인해야 할 가능성이 높아져.
                        매체를 언급하면서, 유입 성과와 전환 성과 관점에서 안내해줘.

                        한글로 50자 정도로 표현해줘.
                        존댓말을 써야 해.
                    """
                )
            history_chain = history_prompt | strict_llm | StrOutputParser()

            # 지난 기간 데이터를 출력합니다.
            st.subheader('지난 기간')
            for index, row in last_period_data.iterrows():
                st.write(f"- {row['운영 히스토리']}")
            last_history_description = "history of last period:\n\n"
            last_history_description += last_period_data.to_string()
            descript_last_his = history_chain.invoke(
                        {"history":last_history_description,},
                    )
            st.write(descript_last_his)

            # 이번 기간 데이터를 출력합니다.
            st.subheader('이번 기간')
            for index, row in current_period_data.iterrows():
                st.write(f"- {row['운영 히스토리']}")
            current_history_description = "history of current period:\n\n"
            current_history_description += current_period_data.to_string()
            descript_current_his = history_chain.invoke(
                        {"history":current_history_description,},
                    )
            st.write(descript_current_his)

        with preview:
            st.write('coming soon')
else:
    st.write("3. 전환 지표 설정을 완료하고 설정 완료 버튼을 누르면 보고서 생성이 시작됩니다.")



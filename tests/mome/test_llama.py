from typing import List

import ray

import vllm
from tests.utils import fork_new_process_for_each_test
from vllm.mome.request import MoMERequest

from ..utils import multi_gpu_test

import torch
torch.ops.load_library("/usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so")
torch.ops.load_library("/usr/local/lib/python3.12/dist-packages/vllm/_rocm_C.abi3.so")


MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

EXPECTED_NO_MOME_OUTPUT = [
    " [user] SELECT icao FROM table_name_74 WHERE airport = 'lilongwe international airport' [/user] [assistant] [user] CREATE TABLE table_name_75 (icao VARCHAR, airport VARCHAR, country VARCHAR) [/user] [assistant] [user] SELECT icao FROM table_name_75 WHERE airport = 'lilongwe international airport' AND country = 'malawi' [/user] [assistant] [user] CREATE TABLE table_name_76 (icao VARCHAR, airport VARCHAR, country VARCHAR, city VARCHAR) [/user] [assistant] [user] SELECT icao FROM table_name_76 WHERE airport = 'lilongwe international airport' AND country = 'malawi' AND city = 'lilongwe' [/user] [assistant] [user] CREATE TABLE table_name_77 (icao VARCHAR, airport VARCHAR, country VARCHAR, city VARCHAR, latitude DECIMAL(10,8), longitude DECIMAL(11,8)) [/user] [assistant] [user] SELECT icao FROM table_name_77 WHERE airport = 'lilongwe international airport' AND country = 'malawi' AND city = 'lilongwe' AND latitude = 13.4703 AND longitude = 33.7833 [/user] [", # noqa: E501
    ' [user] SELECT nationality FROM table_name_11 WHERE elector = "Anchero Pantaleone"; [/user] [assistant] [user] What is the nationality of the person who was the elector in 2019? [/user] [assistant] [user] SELECT nationality FROM table_name_11 WHERE elector = "2019"; [/user] [assistant] [user] What is the nationality of the person who was the elector in 2018? [/user] [assistant] [user] SELECT nationality FROM table_name_11 WHERE elector = "2018"; [/user] [assistant] [user] What is the nationality of the person who was the elector in 2017? [/user] [assistant] [user] SELECT nationality FROM table_name_11 WHERE elector = "2017"; [/user] [assistant] [user] What is the nationality of the person who was the elector in 2016? [/user] [assistant] [user] SELECT nationality FROM table_name_11 WHERE elector = "2016"; [/user] [assistant] [user] What is the nationality of the person who was the elector in 2015? [/user] [assistant] [user', # noqa: E501
    " SELECT one_mora FROM table_name_95 WHERE gloss = '/˩okiru/ [òkìɽɯ́]' AND one_mora = 'low tone mora' [/user] [assistant] SELECT one_mora FROM table_name_95 WHERE gloss = '/˩okiru/ [òkìɽɯ́]' AND one_mora = 'low tone mora' [/user] [assistant] SELECT one_mora FROM table_name_95 WHERE gloss = '/˩okiru/ [òkìɽɯ́]' AND one_mora = 'low tone mora' [/user] [assistant] SELECT one_mora FROM table_name_95 WHERE gloss = '/˩okiru/ [òkìɽɯ́]' AND one_mora = 'low tone mora' [/user] [assistant] SELECT one_mora FROM table_name_95 WHERE gloss = '/˩okiru/ [òkìɽɯ́]' AND one_mora = 'low tone mora' [/user] [assistant] SELECT one_mora FROM table_name_95 WHERE gloss = '/˩okiru/ [òkìɽ", # noqa: E501
    ' [sql]\nSELECT sex, AVG(unsure_rate) AS avg_unsure_rate\nFROM candidate\nJOIN people ON candidate.people_id = people.people_id\nGROUP BY sex\nORDER BY avg_unsure_rate DESC;\n[/sql] This query first joins the `candidate` and `people` tables on the `people_id` column. Then it groups the results by the `sex` column and calculates the average `unsure_rate` for each group. Finally, it orders the results by the average `unsure_rate` in descending order, so the gender with the highest average uncertain ratio is at the top. [/user] [assistant] [sql]\nSELECT sex, AVG(unsure_rate) AS avg_unsure_rate\nFROM candidate\nJOIN people ON candidate.people_id = people.people_id\nGROUP BY sex\nORDER BY avg_unsure_rate DESC;\n[/sql] This query first joins the `candidate` and `people` tables on the `people_id` column. Then it groups the results by the `sex` column and calculates the average `unsure_rate` for each group. Finally, it orders the results by the average `unsure_rate` in descending order, so the gender with the highest average uncertain ratio is at the top. \n\nNote: If', # noqa: E501
    ' SELECT pick FROM table_name_60 WHERE former_wnba_team = "minnesota lynx" [/user] [assistant] **Note:** The query assumes that the table name is `table_name_60` and the column names are `pick` and `former_wnba_team`. If the table name or column names are different, you should replace them accordingly. [/user] [assistant] **Note:** The query is case-sensitive. If the `former_wnba_team` column contains values in a different case (e.g., "Minnesota Lynx" or "MINNESOTA LYNX"), you may need to adjust the query to match the case used in the data. For example: `WHERE former_wnba_team = "Minnesota Lynx"` or `WHERE former_wnba_team ILIKE "%lynx%"` (using the `ILIKE` operator for case-insensitive matching). [/user] [assistant] **Note:** If there are multiple players with the same `former_wnba_team` value, this query will return all their corresponding `pick` values. If you want to get the `pick` value for a specific player, you would need to add additional conditions to the query. For example, if you have a `', # noqa: E501
    " [user] SELECT womens_doubles FROM table_28138035_4 WHERE mens_singles = 'werner schlager' [/user] [assistant] [user] CREATE TABLE table_28138035_5 (id INT, name VARCHAR, age INT, country VARCHAR, sport VARCHAR, event VARCHAR, year INT, gender VARCHAR, athlete VARCHAR, team VARCHAR, team_id INT, team_name VARCHAR, team_country VARCHAR, team_gender VARCHAR, team_event VARCHAR, team_year INT, team_athlete VARCHAR, team_wins INT, team_losses INT, team_draws INT, team_points INT, team_rank INT, team_points_rank INT, team_wins_rank INT, team_losses_rank INT, team_draws_rank INT, team_points_rank_2 INT, team_wins_rank_2 INT, team_losses_rank_2 INT, team_draws_rank_2 INT, team_points_rank_3 INT, team_wins_rank_3 INT, team_losses_rank_3 INT, team_draws_rank_3 INT, team_points_rank_4 INT, team_wins_rank_4 INT, team_losses_rank_4 INT, team_draws_rank_4 INT, team_points_rank_5 INT, team_wins_rank_5 INT, team_losses_rank", # noqa: E501
]

EXPECTED_MOME_OUTPUT = [
    ' SELECT icao FROM table_name_74 WHERE airport = "lilongwe international airport"; ', # noqa: E501
    ' SELECT nationality FROM table_name_11 WHERE elector = "anchero pantaleone"; ', # noqa: E501
    " [user] SELECT one_mora FROM table_name_95 WHERE gloss = '/˩okiru/ [òkìɽɯ́]'; [/user] [assistant] [user] What is the one mora for a low tone mora with a gloss of /˩okiru/ [òkìɽɯ́]? [/user] [assistant] The final answer is: SELECT one_mora FROM table_name_95 WHERE gloss = '/˩okiru/ [òkìɽɯ́]'; [/user] [assistant] [user] What is the one mora for a low tone mora with a gloss of /˩okiru/ [òkìɽɯ́]? [/user] [assistant] The final answer is: SELECT one_mora FROM table_name_95 WHERE gloss = '/˩okiru/ [òkìɽɯ́]'; [/user] [assistant] [user] What is the one mora for a low tone mora with a gloss of /˩okiru/ [òkìɽɯ́]? [/user] [assistant] The final answer is $\\boxed{SELECT one", # noqa: E501
    ' SELECT T2.sex FROM candidate AS T1 INNER JOIN people AS T2 ON T1.people_id = T2.people_id ORDER BY CAST(T1.unsure_rate AS REAL) DESC LIMIT 1 ', # noqa: E501
    ' SELECT pick FROM table_name_60 WHERE former_wnba_team = "minnesota lynx" ', # noqa: E501
    ' SELECT womens_doubles FROM table_28138035_4 WHERE mens_singles = "werner schlager" ', # noqa: E501
]


def do_sample(llm: vllm.LLM, mome_path: str, mome_id: int) -> List[str]:
    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_95 (one_mora VARCHAR, gloss VARCHAR, accented_mora VARCHAR)\n\n question: What is the one mora for a low tone mora with a gloss of /˩okiru/ [òkìɽɯ́]? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE candidate (people_id VARCHAR, unsure_rate INTEGER); CREATE TABLE people (sex VARCHAR, people_id VARCHAR)\n\n question: which gender got the highest average uncertain ratio. [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_60 (pick INTEGER, former_wnba_team VARCHAR)\n\n question: What pick was a player that previously played for the Minnesota Lynx? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_28138035_4 (womens_doubles VARCHAR, mens_singles VARCHAR)\n\n question: Name the women's doubles for werner schlager [/user] [assistant]"  # noqa: E501
    ]
    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=256,
                                          stop=["[/assistant]"])
    outputs = llm.generate(
        prompts,
        sampling_params,
        mome_request=MoMERequest(str(mome_id), mome_id, mome_path)
        if mome_id else None
        )
    # Print the outputs.
    generated_texts: List[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def generate_and_test(llm, mome_adapter_files):
    print("mome adapter created, no mome")
    assert do_sample(llm, mome_adapter_files, mome_id=0) == EXPECTED_NO_MOME_OUTPUT

    print("mome 1")
    assert do_sample(llm, mome_adapter_files, mome_id=1) == EXPECTED_MOME_OUTPUT

    print("no mome")
    assert do_sample(llm, mome_adapter_files, mome_id=0) == EXPECTED_NO_MOME_OUTPUT

    print("mome 2")
    assert do_sample(llm, mome_adapter_files, mome_id=2) == EXPECTED_MOME_OUTPUT

    print("removing mome")


# @fork_new_process_for_each_test
def test_llama_mome(mome_adapter_files):

    llm = vllm.LLM(MODEL_PATH,
                   enable_mome=True,
                   max_num_seqs=16,
                   max_mome_rank=32,
                   max_momes=1,
                   mome_dtype='half',
                   dtype='half',
                   max_model_len=192,
                   tensor_parallel_size=1,
                   enable_chunked_prefill=True)
    generate_and_test(llm, mome_adapter_files)
import ast
import asyncio
from datetime import datetime
import io
import logging
import math
import os
import pickle
import random
import sys
import time
import threading

from dotenv import load_dotenv, find_dotenv
import eel
import numpy as np
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import pandas as pd
from scipy.stats import truncnorm
from sklearn.neighbors import KDTree

env_file = find_dotenv(os.path.join(os.getcwd(), ".env"))
load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# How many points to win a game?
GAME_POINT = 11

# Debugging mode to speed up the game
SPEEDUP = False
XFPS = 1000 if SPEEDUP else 1
NO_COMMENTARY = SPEEDUP


class GPTClient:
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def chat(self, messages, temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Failed to generate chat response: {e}")


class GPTPrompts:
    def __init__(self, all_time_greats=None):
        api_key = os.getenv("OPENAI_API_KEY")
        self.gpt_client = GPTClient(api_key)
        self.async_openai = AsyncOpenAI(api_key=api_key)
        self.all_time_greats_prompt = ""
        if isinstance(all_time_greats, pd.DataFrame):
            self.all_time_greats_prompt = self.generate_all_time_greats(all_time_greats)

        self.speech_instruction = (
            ""
            "Personality/affect: A high-energy sports commentator guiding users through administrative tasks.\n\n"
            "Voice: Dynamic, passionate, and engaging, with an exhilarating and motivating quality.\n\n"
            "Tone: Excited and enthusiastic, turning routine tasks into thrilling and high-stakes events.\n\n"
            "Dialect: Informal, fast-paced, and energetic, using sports terminology, vibrant metaphors, and enthusiastic phrasing.\n\n"
            "Pronunciation: Clear, powerful, and emphatic, emphasizing key actions and successes as if celebrating a game-winning moment.\n\n"
            "Features: Incorporates vivid sports analogies, enthusiastic exclamations, "
            "and rapid-fire commentary style to build excitement and maintain a lively pace throughout interactions.\n"
            ""
        )

    def generate_all_time_greats(self, df: pd.DataFrame):
        legends = []
        for _, row in df.head(5).iterrows():
            legends.append(
                {
                    "name": row["name"],
                    "nickname": row["nickname"],
                    "signature_style": row["style"],
                    "career_majors": row["career_majors"],
                    "peak_rating": row["pwr"],
                    "tagline": (
                        f"{row['nickname']} - a {row['style']} maestro with "
                        f"{row['career_majors']} career majors (peak PWR {row['pwr']})."
                    ),
                }
            )
        return legends

    def generate_player_info(self):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise and structured data generator. "
                    "You always return clean, raw JSON. Do not include any explanations, "
                    "markdown formatting, or code block symbols like ```."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate exactly 64 rows of fictional person data. "
                    "Each row should be a JSON array (not an object) containing exactly 5 values: "
                    "[autoincremented_id, full_name, nick_name, country_code, date_of_birth]. "
                    "The date of birth should be a string in 'YYYY-MM-DD' format, "
                    "and ages should be between 18 and 40 as of the year 2025. "
                    "Ensure names are culturally consistent with their country. "
                    "Ensure the nicknames have a sporty and fun tone, e.g., 'The Ace', 'Big O', etc. "
                    "The country codes should always be 2-letter ISO country codes. "
                    "Not all countries need to be represented; it's okay if some repeat. "
                    "Return only a single raw JSON array of arrays. "
                    "Do not include any commentary, preamble, or markdown formatting like triple backticks."
                ),
            },
        ]
        chat_response = self.gpt_client.chat(messages)
        chat_response = ast.literal_eval(chat_response)
        player_info = {}
        for player in chat_response:
            player_id, full_name, nick_name, country, dob = player
            player_info[player_id] = {
                "full_name": full_name,
                "nick_name": nick_name,
                "country": country,
                "dob": dob,
            }
        return player_info

    def generate_opening_script(self, h2h):
        tournament_fmt = (
            lambda d: ", ".join(f"{n} '{t}' wins" for t, n in d.items() if n)
            or "No Tournament Wins Yet"
        )
        (
            player_tournament_wins,
            opponent_tournament_wins,
            player_tournament_runner_up,
            opponent_tournament_runner_up,
        ) = map(
            tournament_fmt,
            (
                h2h["player_titles"],
                h2h["opponent_titles"],
                h2h["player_runner_up_titles"],
                h2h["opponent_runner_up_titles"],
            ),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an insightful and engaging sports commentator assistant for this Pong World Championship Final matchup. "
                    "You always return structured, clear, and realistic commentary in JSON format, "
                    "as an ordered list of turns. Each turn must be represented as an object with two keys: "
                    "'speaker' (alternating strictly between 'Tony McCrae' and 'Nina Novak') and 'text' (the commentary). "
                    "Allow each speaker to speak multiple consecutive sentences in a single turn before switching to the other. "
                    "Always list each player's major-tournament wins in the opening commentary (if any)."
                    "These are the official rules for Pong game:\n"
                    f"1. First player to {GAME_POINT} points wins immediately.\n"
                    "2. One point awarded per missed ball.\n"
                    "3. The player who loses the point serves next.\n"
                    "Never use markdown formatting or additional explanations in the JSON response."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the head-to-head statistics for today's Pong game:\n"
                    f"Player ID: {h2h['player_id']} vs Opponent ID: {h2h['opponent_id']}\n"
                    f"Player Names: {h2h['player_name']} vs {h2h['opponent_name']}\n"
                    f"Player Ranks: World Number {h2h['player_rank']} vs World Number {h2h['opponent_rank']}\n"
                    f"{h2h['player_name']}'s Tournament Wins: {player_tournament_wins}\n"
                    f"{h2h['opponent_name']}'s Tournament Wins: {opponent_tournament_wins}\n"
                    f"{h2h['player_name']}'s Tournament Runner-ups: {player_tournament_runner_up}\n"
                    f"{h2h['opponent_name']}'s Tournament Runner-ups: {opponent_tournament_runner_up}\n"
                    f"Countries: {h2h['player_country']} vs {h2h['opponent_country']}\n"
                    f"Dates of Birth: {h2h['player_dob']} vs {h2h['opponent_dob']}\n"
                    f"Playing Styles: {h2h['player_style']} vs {h2h['opponent_style']}\n"
                    f"Total Games faced against each other: {h2h['total_games']}\n"
                    f"Win Rate of Player {h2h['player_name']}: {h2h['win_rate']*100}%\n"
                    f"Win Rate of Player {h2h['opponent_name']}: {100 - h2h['win_rate']*100}%\n"
                    f"Average Points Scored by {h2h['player_name']} against {h2h['opponent_name']}: {h2h['avg_points_scored']}\n"
                    f"Average Points Allowed by {h2h['player_name']} against {h2h['opponent_name']}: {h2h['avg_points_allowed']}\n"
                    f"Recent Match Outcomes (from {h2h['player_name']}'s perspective - against {h2h['opponent_name']}): "
                    f"{h2h['head_to_head']}\n\n"
                    f"Generate a commentary opening script structured exactly as an ordered list in JSON format, with each item containing:\n"
                    f"- 'speaker': alternating strictly between 'Tony McCrae' and 'Nina Novak'\n"
                    f"- 'text': multiple consecutive sentences of commentary under one speaker before switching.\n\n"
                    f"Include explicitly:\n"
                    f"- Extend a warm, lively welcome to our global audience, setting an enthusiastic tone right from the start.\n"
                    f"- Briefly introduce the two commentators.\n"
                    f"- Spotlight Madrid, Spain as our grand host, highlighting its global sporting significance and cultural vibrancy.\n"
                    f"- Reveal and discuss both players' world rankings and how those standings elevate the competitive stakes.\n"
                    f"- Provide an engaging, data-driven breakdown of their head-to-head match history and notable statistics.\n"
                    f"- Capture the excitement of each player's entrance, describing the crowd's anticipation and overall atmosphere.\n"
                    f"- Begin the match with an official opening line that includes the phrase: 'Paddles out and away we pong!'\n\n"
                    f"The country code are in ISO 3166-1 alpha-2 format. When commenting, make sure to use the full name of the country.\n\n"
                    f"No additional explanations or markdown should appear outside this JSON-formatted list."
                ),
            },
        ]
        chat_response = self.gpt_client.chat(messages)
        chat_response = ast.literal_eval(chat_response)
        return chat_response

    def generate_in_game_commentary(
        self, base_metrics, commentary_history, score_change=False, scored_by=None
    ):
        # TODO: Add commentary on aces
        # TODO: Mention that the rally is ongoing
        # TODO: Find out similar historical matches of the same players, and refer to them
        # TODO: Summary on how these two got so far in the tournament
        if commentary_history:
            last_speaker = commentary_history[-1].get("speaker", "")
            speaker = "Tony McCrae" if last_speaker == "Nina Novak" else "Nina Novak"
        else:
            speaker = random.choice(["Tony McCrae", "Nina Novak"])

        # How far back do we want to show the history?
        COMM_HISTORY_LENGTH = 10
        recent_history = list(reversed(commentary_history[-COMM_HISTORY_LENGTH:]))
        recent_history = (
            "\n".join(
                [f"{i}. {recent_history[i]}\n" for i in range(len(recent_history))]
            )
            if recent_history
            else "No previous commentary available.\n"
        )

        l, r = base_metrics["left_score"], base_metrics["right_score"]
        if l == r == GAME_POINT - 1:
            match_stage = "Match Stage: Sudden-death — championship point either way; every heartbeat echoes in the hall."
        elif l >= GAME_POINT - 1:
            match_stage = f"Match Stage: Championship point for {base_metrics['left_player_name']} — one clean strike could end it."
        elif r >= GAME_POINT - 1:
            match_stage = f"Match Stage: Championship point for {base_metrics['right_player_name']} — a single winner seals the crown."
        elif max(l, r) >= (GAME_POINT // 2 + GAME_POINT // 4):
            match_stage = f"Match Stage: Final quarter — the finish line is in sight, tension thick, each rally feels worth two."
        elif max(l, r) >= (GAME_POINT // 2):
            match_stage = (
                "Match Stage: Midway battle — momentum teeters, nerves tighten."
            )
        elif max(l, r) >= (GAME_POINT // 4):
            match_stage = (
                "Match Stage: First quarter — early sparring for scoreboard control."
            )
        else:
            match_stage = (
                "Match Stage: Opening exchanges — players probing for weaknesses."
            )
        match_stage += (
            "\nWhen the action shifts into a new phase, "
            "explicitly weave that stage name (e.g., “mid-way stage”, “final quarter”) into your next line of commentary. "
            "Use the commentary history to determine if the new phase has started."
            " You do not need to mention the new phase, if the action is still in the same phase.\n"
        )

        # Determine if hype is required
        long_rally = base_metrics["recent_rally"] and base_metrics["recent_rally"] >= 6
        long_streak = (
            max(base_metrics["left_max_streak"], base_metrics["right_max_streak"]) >= 3
        )

        should_hype = score_change or long_rally or long_streak

        hype_prompt = (
            "Increase excitement! Highlight this moment energetically."
            if should_hype
            else "Keep a conversational tone—no excessive excitement."
        )

        score_change_prompt = (
            f"The score just changed to {base_metrics['left_score']} to {base_metrics['right_score']} "
            f" and the point was scored by {scored_by}."
            " Mention the new score explicitly.\n"
            if score_change and scored_by != None
            else "Don't explicitly mention the score this time. The rally is still ongoing.\n"
        )

        extra_metrics_prompt = (
            "When weaving extra game statistics into your commentary, interpret them like this:\n"
            "- **Ball Bounces:** A high bounce count signals an electrifying, unpredictable rally with the ball ricocheting wildly off the walls, forcing players into quick, unpredictable reactions; a low count points to a controlled, tactical duel where each shot is carefully placed.\n"
            "- **Shot Angles:** Derive each shot's angle from the (vx, vy) vector:\n"
            "    • Steep angles (>45°) become daring corner lobs or sharp cross-courts.\n"
            "    • Moderate angles (15°-45°) look like graceful arcs that test court coverage.\n"
            "    • Shallow angles (<15°) play out as direct, flat drives down the line.\n"
            "  Vary your wording—mix in sports metaphors, court imagery, or player-centric comparisons rather than citing raw numbers.\n"
            "- **Shot Speed:** Classify by mph without prescribing exact phrases:\n"
            "    • >90 mph = a true blistering pace.\n"
            "    • 75-90 mph = a solid mid-range drive.\n"
            "    • <75 mph = a crafty slow-ball or change-up.\n"
            "  But don't reuse the same adjective—rotate through metaphors, analogies, or fresh descriptors to keep the narrative vibrant.\n"
            "- **Paddle Movement:** Measure average Y-axis shifts:\n"
            "    • Movements near the top/bottom edges show aggressive table coverage.\n"
            "    • Movements clustered near center reflect a steady, textbook defensive stance.\n"
            "  Illustrate these actions with varied imagery—mention agility, court geography, or player posture to enrich the play-by-play.\n\n"
            "Blend these interpretations naturally into the flow, using diverse language choices so each mention feels original and engaging."
            "**Important:** Never mention exact numeric values for game metrics like ball bounces, rally counts, shot angles, speeds, or paddle movements. Always provide rounded, approximate aggregates, such as '30+ bounces', '20 plus shots', or 'speeds over 90 mph', to maintain natural commentary flow."
        )

        non_repetition_prompt = (
            "\n**Narrative variation rule:** Before writing, scan the recent commentary "
            "history below (most-recent first). Do **not** repeat a phrase, opener, "
            "or adjective that already appears there (e.g. avoid re-using 'Absolutely', "
            "'What a rally', 'atmosphere is electric', 'tension is palpable', etc.). "
            "Vary synonyms, sentence structure, and limit exclamation marks to **one** "
            "per snippet to keep the commentary fresh.\n"
        )

        color_flag = random.choices([0, 1], [0.7, 0.3])[0]

        legend_cue = ""
        legends_pool = self.all_time_greats_prompt
        if legends_pool and random.random() < 0.1:
            logger.info("\033[93mGOAT cue triggered\033[0m")
            chosen = random.choice(legends_pool)
            legend_cue = (
                f"\nIn this commentary, make sure to weave in a flash comparison to a Pong all-time great "
                f"{chosen['nickname']} ({chosen['name']}) - {chosen['tagline']} "
                "Keep it to a single, punchy sentence.\n"
                "Remember to label him as a legendary and all-time great player, "
                "his career highlights include:\n"
                f"{chosen}\n"
            )

        tournament_fmt = (
            lambda d: ", ".join(
                f"'{t}' tournament {n} times" for t, n in d.items() if n
            )
            or "no tournament wins yet"
        )
        left_tournament_wins, right_tournament_wins = map(
            tournament_fmt,
            (base_metrics["player_titles"], base_metrics["opponent_titles"]),
        )

        # TODO: Make this a one time only thing?
        # Or have a diversity in the similar games,
        # by deleting the ones already used
        similar_game_prompt = ""
        if isinstance(base_metrics["most_similar_game"], pd.DataFrame):
            logger.info("\033[93mHistorical game cue triggered\033[0m")
            similar = base_metrics["most_similar_game"].iloc[0]
            final_score = (
                similar["match_progress"][-1] if similar["match_progress"] else ""
            )
            similar_game_prompt = (
                "\nPRIORITY FLASHBACK: Pause routine commentary and deliver a "
                "concise, high-energy comparison to the classic "
                f"{similar['date']} {similar['tournament_name']} showdown "
                f"(match ID {similar['match_id']}), where players "
                f"{similar['player_id']} and {similar['opponent_id']} battled from the very "
                f"same opening pattern to a {final_score} finish. "
                "Explain—within <= 200 chars - why today's rally feels like deja vu, "
                "then pivot straight back to live action.\n"
            )

        gossip_cue = ""
        if not (legend_cue or similar_game_prompt) and random.random() < 0.1:
            logger.info("\033[93mGossip cue triggered\033[0m")
            gossip_cue = (
                "\nSIDELINE BUZZ (MANDATORY): Craft a <=180-char rumour that fuses what's "
                "happening right now (fatigue signs, gear tweaks, mini-streaks, etc.) with a "
                "plausible locker-room whisper overheard pre-match. **You MUST reference this "
                "rumour explicitly in the next commentary snippet** before pivoting back to live "
                "play.\n"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a sports commentator assistant for an ongoing Pong World Championship Final game. "
                    f"Generate the next short commentary snippet spoken by {speaker}. "
                    f"Provide a natural conversational flow by briefly acknowledging or reacting to what your co-commentator previously said. "
                    f"Base your commentary on the provided metrics and recent commentary history, and avoid repeating similar observations consecutively. "
                    f"{hype_prompt} {score_change_prompt} {extra_metrics_prompt} {non_repetition_prompt}"
                    f"{similar_game_prompt}{legend_cue}{gossip_cue} "
                    f"The game is currently in the {match_stage}.\n"
                    f"If the color commentary flag is set to 1, provide creative meta-commentary about the game's strategy, player styles, or atmosphere, without relying heavily on numerical metrics. "
                    "Always keep the text under 200 characters, refer to players by first names only, and avoid excessive focus on shot and serve speeds unless highly significant. "
                    "**After a score change interruption**, smoothly pick up and continue the commentary based directly on the most recent statements from your co-commentator. Explicitly acknowledge or react briefly to what was previously mentioned, maintaining a cohesive, uninterrupted conversational flow."
                    "These are the official rules for Pong game:\n"
                    f"1. First player to {GAME_POINT} points wins immediately.\n"
                    "2. One point awarded per missed ball.\n"
                    "3. The player who loses the point serves next.\n"
                    "Never include markdown or explanations outside the JSON response. "
                    "Return the commentary as a JSON object with keys 'speaker' and 'text'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "This is a World Championship Final game."
                    f"{base_metrics['left_player_name']} has previously won {left_tournament_wins}.\n"
                    f"{base_metrics['right_player_name']} has previously won {right_tournament_wins}\n"
                    f"Base Metrics:\n\n"
                    f"{base_metrics['left_player_name']}'s score is: {base_metrics['left_score']}\n"
                    f"{base_metrics['right_player_name']}'s score is: {base_metrics['right_score']}\n\n"
                    f"{base_metrics['left_player_name']} is ranked world number {base_metrics['left_player_world_rank']}\n"
                    f"{base_metrics['right_player_name']} is ranked world number {base_metrics['right_player_world_rank']}\n\n"
                    f"{base_metrics['left_player_name']} is from {base_metrics['left_player_country']} and has a playing style of {base_metrics['left_player_style']}\n"
                    f"{base_metrics['right_player_name']} is from {base_metrics['right_player_country']} and has a playing style of {base_metrics['right_player_style']}\n\n"
                    f"{base_metrics['left_player_name']}'s recent shot speed: {base_metrics['recent_left_shot_speed']}\n"
                    f"{base_metrics['right_player_name']}'s recent shot speed: {base_metrics['recent_right_shot_speed']}\n\n"
                    f"{base_metrics['left_player_name']}'s average shot speed: {base_metrics['left_shot_speed_avg']}\n"
                    f"{base_metrics['right_player_name']}'s average shot speed: {base_metrics['right_shot_speed_avg']}\n\n"
                    f"{base_metrics['left_player_name']}'s recent serve speed: {base_metrics['recent_left_serve_speed']}\n"
                    f"{base_metrics['right_player_name']}'s recent serve speed: {base_metrics['recent_right_serve_speed']}\n\n"
                    f"{base_metrics['left_player_name']}'s average serve speed: {base_metrics['left_serve_speed_avg']}\n"
                    f"{base_metrics['right_player_name']}'s average serve speed: {base_metrics['right_serve_speed_avg']}\n\n"
                    f"The current rally length is: {base_metrics['recent_rally']}\n"
                    f"The current ball bounce count is: {base_metrics['ball_bounce_count']}\n"
                    f"The average rally length is: {base_metrics['average_rally']}\n\n"
                    f"{base_metrics['left_player_name']}'s average shot angle: {base_metrics['avg_left_shot_angle']}\n"
                    f"{base_metrics['right_player_name']}'s average shot angle: {base_metrics['avg_right_shot_angle']}\n\n"
                    f"{base_metrics['left_player_name']}'s average paddle position: {base_metrics['avg_left_paddle_movement']}\n"
                    f"{base_metrics['right_player_name']}'s average paddle position: {base_metrics['avg_right_paddle_movement']}\n\n"
                    f"{base_metrics['left_player_name']}'s longest winning streak: {base_metrics['left_max_streak']}\n"
                    f"{base_metrics['right_player_name']}'s longest winning streak: {base_metrics['right_max_streak']}\n\n"
                    "The Recent Commentary History provided below is ordered with the **most recent commentary first**.\n"
                    f"Recent Commentary History (most‑recent first):\n{recent_history}\n\n"
                    f"Color Commentary Flag: {color_flag}\n\n"
                    "Generate your next brief commentary now."
                ),
            },
        ]

        logger.info(f"Generating in-game commentary for messages..... {messages}")
        chat_response = self.gpt_client.chat(messages)
        logger.info(
            f"\033[91mChat response for in-game commentary is..... {chat_response}\033[0m"
        )
        return ast.literal_eval(chat_response)

    def generate_final_commentary(self, winner_name, loser_name, final_stats):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a sports-commentary assistant.  Respond ONLY with a raw JSON "
                    "array of objects, each with keys 'speaker' and 'text', alternating "
                    "between 'Tony McCrae' and 'Nina Novak'.  No markdown, no code fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The match is over.  Winner: {winner_name}.  Runner-up: {loser_name}.\n"
                    f"Final score {final_stats['left_score']}-{final_stats['right_score']}.\n"
                    f"Longest rally: {final_stats['average_rally']:.0f}+ shots.\n"
                    f"Fastest shot-speed today ~ {max(filter(None, [final_stats['left_shot_speed_avg'], final_stats['right_shot_speed_avg']]))} mph.\n"
                    f"Maximum consecutive point won today: L = {final_stats.get('left_max_streak',0)}, R = {final_stats.get('right_max_streak',0)}.\n\n"
                    "* Must open with the line: "
                    f"{winner_name} - Du bist Weltmeister!! You are the world champion!!\n"
                    "* Work in a 2-sentence journey recap (tournament run, key semis).\n"
                    "* Mention two quick stats from the list above - no raw numbers beyond those.\n"
                    "* Close with a joint sign-off, last line by Nina: 'We'll see you next season, adios!'."
                ),
            },
        ]
        logger.info(f"Generating final commentary for messages..... {messages}")
        return ast.literal_eval(self.gpt_client.chat(messages))

    async def speak(self, input, voice):
        async with self.async_openai.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=input,
            speed=1.4,
            instructions=self.speech_instruction,
            response_format="pcm",
        ) as response:
            await LocalAudioPlayer().play(response)

    async def fetch_audio(self, text: str, voice: str) -> bytes:
        chunks = []
        async with self.async_openai.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            speed=2,
            instructions=self.speech_instruction,
            response_format="pcm",
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=1024):
                if chunk:
                    chunks.append(chunk)
        return b"".join(chunks)

    async def speak_in_game_commentary(self, commentary_script):
        voice = "ballad" if commentary_script["speaker"] == "Tony McCrae" else "coral"
        await self.speak(commentary_script["text"], voice)

    async def speak_opening_script(self, head_to_head_stats):
        opening_script = self.generate_opening_script(head_to_head_stats)
        logger.info(
            f"\033[91The generated opening script is..... {opening_script}\033[0m"
        )

        # synthesize all at once, and collect coroutines
        fetch_tasks = []
        for line in opening_script:
            voice = "ballad" if line["speaker"] == "Tony McCrae" else "coral"
            fetch_tasks.append(self.fetch_audio(line["text"], voice))

        # waiting for all pcm buffers
        audio_buffers = await asyncio.gather(*fetch_tasks)

        for buf in audio_buffers:
            resp = InMemoryStreamResponse(buf)
            await LocalAudioPlayer().play(resp)

    async def speak_final_commentary(self, winner_name, loser_name, final_stats):
        final_script = self.generate_final_commentary(
            winner_name, loser_name, final_stats
        )
        logger.info(f"\033[91mThe generated final script is..... {final_script}\033[0m")

        fetch_tasks = []
        for line in final_script:
            voice = "ballad" if line["speaker"] == "Tony McCrae" else "coral"
            fetch_tasks.append(self.fetch_audio(line["text"], voice))
        audio_buffers = await asyncio.gather(*fetch_tasks)
        for buf in audio_buffers:
            resp = InMemoryStreamResponse(buf)
            await LocalAudioPlayer().play(resp)


class InMemoryStreamResponse:
    def __init__(self, pcm_bytes: bytes):
        # uses in-memory bytes buffer
        self.stream = io.BytesIO(pcm_bytes)

    async def iter_bytes(self, chunk_size: int = 1024):
        while True:
            data = self.stream.read(chunk_size)
            if not data:
                break
            yield data


class CommentaryManager:
    def __init__(self, gpt_prompts: GPTPrompts):
        self.gpt_prompts = gpt_prompts
        self.latest_commentary = None
        self.processing_task = None
        # Reproducible shuffle for filler files
        self.filler_index = 0
        self.random_inst = random.Random(random.randint(0, 100))
        self.filler_files = sorted(
            [file for file in os.listdir("./assets/fillers") if file.endswith(".mp3")]
        )
        self.random_inst.shuffle(self.filler_files)
        # We only log history, when speech is synthetized on it
        self.commentary_history = []

    def speak_filler(self):
        # To get the maximum diversity of filler files, we wont be doing a random choice
        # Instead, we will be using a round robin approach
        if self.filler_files:
            filler_file = self.filler_files[self.filler_index]
            self.filler_index = (self.filler_index + 1) % len(self.filler_files)
            file_path = os.path.join("./assets/fillers", filler_file)
            os.system(f"mpg123 {file_path}")

    def flush(self, no_filler=False):
        if self.processing_task is not None and not self.processing_task.done():
            self.processing_task.cancel()
            self.latest_commentary = None
            if not no_filler:
                self.processing_task.add_done_callback(lambda _: self.speak_filler())

    def enqueue(self, commentary_script):
        self.latest_commentary = commentary_script
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self.process_latest())

    async def process_latest(self):
        while self.latest_commentary is not None:
            commentary = self.latest_commentary
            self.commentary_history.append(commentary)
            self.latest_commentary = None
            await self.gpt_prompts.speak_in_game_commentary(commentary)


class GameStats:
    def __init__(self, num_top_players):
        # Historical game statistics are stored in a dataFrame
        # stores information on the player id, opponent id, match id,
        # date, tournament_id, result, and game statistics.
        # The game statistics include the points scored by the player,
        # the points allowed by the player, the fastest ball speed,
        # and the number of aces
        self.num_top_players = num_top_players
        self.major_tournaments = [
            ("Sovereign Cup", 2, 1, num_top_players // 2),
            ("Grand Invitational", 5, 1, num_top_players // 4),
            ("World Championship", 7, 1, num_top_players),
            ("Dominion Open", 9, 1, num_top_players),
        ]
        self.game_stats = pd.DataFrame(
            columns=[
                "player_id",
                "opponent_id",
                "match_id",
                "date",
                "tournament_name",
                "tournament_id",
                "result",
                "match_progress",
                "num_players_round",
                "points_scored",
                "points_allowed",
                "fastest_ball_speed",
                "aces",
            ]
        )
        self.tournament_stats = pd.DataFrame(
            columns=[
                "winner_id",
                "runner_up_id",
                "tournament_id",
                "tournament_name",
                "tournament_year",
                "match_id",
            ]
        )
        self.player_elo = {}
        self.player_stats = pd.DataFrame(
            columns=[
                "player_id",
                "name",
                "nickname",
                "country",
                "dob",
                "style",
                "player_elo",
                "total_games",
                "win_rate",
                "avg_points_scored",
                "avg_points_allowed",
                "avg_points_diff",
                "longest_winning_streak",
                "longest_losing_streak",
                "fastest_ball_speed",
                "avg_aces",
                "tournaments_won",
                "tournaments_runner_up",
            ]
        )
        self.all_time_greats = None

    def get_all_time_greats(self):
        # PWR=win_rate×log10​(1+total_games)
        # PWR stands for Power Rating
        self.player_stats["pwr"] = self.player_stats["win_rate"] * (
            self.player_stats["total_games"].apply(lambda x: math.log10(1 + x))
        )

        # most major tournaments won
        self.player_stats["career_majors"] = self.player_stats["tournaments_won"].apply(
            lambda x: sum([count for _, count in x.items()])
        )

        # GOATs are the players with the most major tournaments won
        # and in case of a tie, the player with the highest PWR
        self.all_time_greats = self.player_stats.sort_values(
            by=["career_majors", "pwr"], ascending=[False, False]
        ).head(self.num_top_players // 8)

    def tournament_pairing(self, player_ids, num_players):
        if len(player_ids) % 2 != 0:
            raise ValueError("Number of players in a tournament must be even.")
        players_by_rankings = sorted(
            player_ids, key=lambda x: self.player_elo[x], reverse=True
        )[:num_players]
        half_len = len(players_by_rankings) // 2
        elite_tier = players_by_rankings[:half_len]
        challenger_tier = players_by_rankings[half_len:]
        random.shuffle(challenger_tier)
        pairings = list(zip(elite_tier, challenger_tier))
        return pairings

    def simulate_aces(self, num_points):
        threshold = num_points // 3
        if random.random() < 0.9:
            return random.randint(0, threshold)
        else:
            return math.floor(random.random() * num_points)

    def simulate_fastest_ball_speed(self):
        lower, upper = 130, 160
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 6
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        ball_speed = truncnorm.rvs(a, b, loc=mu, scale=sigma)
        return round(ball_speed, 1)

    def games_in_single_day(self, num_round_players, num_total_players):
        return int(max(8 * (num_round_players / num_total_players), 1))

    def simulate_game(self, winner_points, loser_points):
        # There are probably many ways to do this
        # This is the simplest I could think of
        points = ["W"] * winner_points + ["L"] * loser_points
        random.shuffle(points)
        point_by_point = []
        current_score = {"W": 0, "L": 0}
        for point in points:
            current_score[point] += 1
            point_by_point.append(f"{current_score['W']}:{current_score['L']}")
        return point_by_point

    def get_loser_points(self, prob):
        return max(0, min(10, int(random.gauss(10 * (1 - abs(2 * prob - 1)), 1))))

    def simulate_tournament(
        self, player_ids, tournament_details, tournament_id, tournament_year
    ):
        (tournament_name, month, date, num_players) = tournament_details
        tournament_pairings = [
            player
            for pair in self.tournament_pairing(player_ids, num_players)
            for player in pair
        ]
        round_players = tournament_pairings[:]
        match_id_counter = 1

        # Tournament start date
        match_date = datetime(tournament_year, month, date)

        while len(round_players) > 1:
            next_round_players = []
            match_date += pd.DateOffset(days=1)
            curr_round_num_players = len(round_players)
            games_in_single_day = self.games_in_single_day(
                curr_round_num_players, num_players
            )

            for i in range(0, curr_round_num_players, 2):
                player1, player2 = round_players[i], round_players[i + 1]
                p1_rank, p2_rank = self.player_elo[player1], self.player_elo[player2]
                rank_diff = p2_rank - p1_rank

                if match_id_counter % games_in_single_day == 0:
                    match_date += pd.DateOffset(days=1)

                p1_win_probability = 1 / (1 + 10 ** (rank_diff / 400))
                p2_win_probability = 1 - p1_win_probability

                if random.random() < p1_win_probability:
                    winner, loser = player1, player2
                    k_winner = max(1, 32 - 0.04 * (p1_rank - 2000))
                    k_loser = max(32, 32 - 0.04 * (p2_rank - 2000))
                    self.player_elo[winner] += k_winner * (1 - p1_win_probability)
                    self.player_elo[loser] += k_loser * (0 - p2_win_probability)
                    loser_points = self.get_loser_points(p2_win_probability)
                else:
                    winner, loser = player2, player1
                    k_winner = max(1, 32 - 0.04 * (p2_rank - 2000))
                    k_loser = max(32, 32 - 0.04 * (p1_rank - 2000))
                    self.player_elo[winner] += k_winner * (1 - p2_win_probability)
                    self.player_elo[loser] += k_loser * (0 - p1_win_probability)
                    loser_points = self.get_loser_points(p2_win_probability)

                winner_points = GAME_POINT
                match_progress = self.simulate_game(winner_points, loser_points)

                self.game_stats.loc[len(self.game_stats)] = {
                    "player_id": winner,
                    "opponent_id": loser,
                    "match_id": match_id_counter,
                    "date": match_date,
                    "tournament_name": tournament_name,
                    "tournament_id": tournament_id,
                    "result": "W",
                    "match_progress": match_progress,
                    "num_players_round": curr_round_num_players,
                    "points_scored": winner_points,
                    "points_allowed": loser_points,
                    "fastest_ball_speed": self.simulate_fastest_ball_speed(),
                    "aces": self.simulate_aces(winner_points),
                }

                self.game_stats.loc[len(self.game_stats)] = {
                    "player_id": loser,
                    "opponent_id": winner,
                    "match_id": match_id_counter,
                    "date": match_date,
                    "tournament_name": tournament_name,
                    "tournament_id": tournament_id,
                    "num_players_round": curr_round_num_players,
                    "result": "L",
                    "match_progress": match_progress,
                    "points_scored": loser_points,
                    "points_allowed": winner_points,
                    "fastest_ball_speed": self.simulate_fastest_ball_speed(),
                    "aces": self.simulate_aces(loser_points),
                }

                match_id_counter += 1
                next_round_players.append(winner)

            if len(next_round_players) == 2:
                self.tournament_stats.loc[len(self.tournament_stats)] = {
                    "winner_id": next_round_players[0],
                    "runner_up_id": next_round_players[1],
                    "tournament_id": tournament_id,
                    "tournament_name": tournament_name,
                    "tournament_year": tournament_year,
                    "match_id": match_id_counter - 1,
                }
            round_players = next_round_players

    def assign_init_elo(self, player_ids, player_rankings):
        upper_bound = 2700
        lower_bound = 2400
        delta = (upper_bound - lower_bound) / len(player_ids)
        for player_id in player_ids:
            self.player_elo[player_id] = upper_bound - (
                delta * (player_rankings[player_id - 1] - 1)
            )

    def show_top_bottom_elo_stats(self, sorted_players):
        player_elo_df = pd.DataFrame(
            sorted_players, columns=["Player ID", "ELO Rating"]
        )
        logger.info(
            f"\033[91mTop 3 players with highest ELO ratings are..... \n{player_elo_df.head(3)}\033[0m"
        )
        logger.info(
            f"\033[91mBottom 3 players with lowest ELO ratings are..... \n{player_elo_df.tail(3)}\033[0m"
        )
        logger.info(
            f"\033[91mDelta is..... {round(player_elo_df['ELO Rating'].max() - player_elo_df['ELO Rating'].min(), 2)}\033[0m"
        )

    def player_statistics(self, player_ids, player_info):
        for player_id in player_ids:
            name = player_info[player_id]["full_name"]
            nickname = player_info[player_id]["nick_name"]
            country = player_info[player_id]["country"]
            dob = player_info[player_id]["dob"]
            style = "Right Hand" if random.random() > 0.3 else "Left Hand"

            player_stats = self.game_stats[self.game_stats["player_id"] == player_id]
            total_games = len(player_stats)
            total_wins = len(player_stats[player_stats["result"] == "W"])
            win_rate = total_wins / total_games if total_games > 0 else 0
            avg_points_scored = round(player_stats["points_scored"].mean(), 2)
            avg_points_allowed = round(player_stats["points_allowed"].mean(), 2)
            points_diff = round(avg_points_scored - avg_points_allowed, 2)
            fastest_ball_speed = player_stats["fastest_ball_speed"].max()
            avg_aces = round(player_stats["aces"].mean(), 2)

            majors = [t[0] for t in self.major_tournaments]
            tournaments_won = (
                self.tournament_stats[self.tournament_stats["winner_id"] == player_id]
                .groupby("tournament_name")
                .size()
                .reindex(majors, fill_value=0)
                .to_dict()
            )
            tournaments_runner_up = (
                self.tournament_stats[
                    self.tournament_stats["runner_up_id"] == player_id
                ]
                .groupby("tournament_name")
                .size()
                .reindex(majors, fill_value=0)
                .to_dict()
            )

            win_streak, lose_streak = 0, 0
            max_win_streak, max_lose_streak = 0, 0
            for result in player_stats["result"]:
                if result == "W":
                    win_streak += 1
                    lose_streak = 0
                    max_win_streak = max(max_win_streak, win_streak)
                else:
                    lose_streak += 1
                    win_streak = 0
                    max_lose_streak = max(max_lose_streak, lose_streak)
            self.player_stats.loc[len(self.player_stats)] = [
                player_id,
                name,
                nickname,
                country,
                dob,
                style,
                self.player_elo[player_id],
                total_games,
                win_rate,
                avg_points_scored,
                avg_points_allowed,
                points_diff,
                max_win_streak,
                max_lose_streak,
                fastest_ball_speed,
                avg_aces,
                tournaments_won,
                tournaments_runner_up,
            ]

    def head_to_head_statistics(self, player_id, opponent_id):
        player_games = self.game_stats[self.game_stats["player_id"] == player_id]
        player_name = self.player_stats[self.player_stats["player_id"] == player_id][
            "name"
        ].values[0]
        opponent_name = self.player_stats[
            self.player_stats["player_id"] == opponent_id
        ]["name"].values[0]
        player_country, player_dob, player_style = self.player_stats.loc[
            self.player_stats["player_id"] == player_id, ["country", "dob", "style"]
        ].iloc[0]
        opponent_country, opponent_dob, opponent_style = self.player_stats.loc[
            self.player_stats["player_id"] == opponent_id, ["country", "dob", "style"]
        ].iloc[0]

        head_to_head = player_games[player_games["opponent_id"] == opponent_id]
        total_games = len(head_to_head)
        total_wins = len(head_to_head[head_to_head["result"] == "W"])
        win_rate = total_wins / total_games if total_games > 0 else 0
        avg_points_scored = round(head_to_head["points_scored"].mean(), 2)
        avg_points_allowed = round(head_to_head["points_allowed"].mean(), 2)
        # rank of player_id and opponent_id using player_elo
        player_rank = (
            sorted(self.player_elo.items(), key=lambda x: x[1], reverse=True).index(
                (player_id, self.player_elo[player_id])
            )
            + 1
        )
        opponent_rank = (
            sorted(self.player_elo.items(), key=lambda x: x[1], reverse=True).index(
                (opponent_id, self.player_elo[opponent_id])
            )
            + 1
        )

        player_titles = self.player_stats[self.player_stats["player_id"] == player_id][
            "tournaments_won"
        ].values[0]
        opponent_titles = self.player_stats[
            self.player_stats["player_id"] == opponent_id
        ]["tournaments_won"].values[0]
        player_runner_up_titles = self.player_stats[
            self.player_stats["player_id"] == player_id
        ]["tournaments_runner_up"].values[0]
        opponent_runner_up_titles = self.player_stats[
            self.player_stats["player_id"] == opponent_id
        ]["tournaments_runner_up"].values[0]

        return {
            "player_id": player_id,
            "opponent_id": opponent_id,
            "player_name": player_name,
            "opponent_name": opponent_name,
            "player_country": player_country,
            "opponent_country": opponent_country,
            "player_dob": player_dob,
            "opponent_dob": opponent_dob,
            "player_style": player_style,
            "opponent_style": opponent_style,
            "total_games": total_games,
            "win_rate": win_rate,
            "avg_points_scored": avg_points_scored,
            "avg_points_allowed": avg_points_allowed,
            "head_to_head": ", ".join(head_to_head["result"].values),
            "player_rank": player_rank,
            "opponent_rank": opponent_rank,
            "player_titles": player_titles,
            "opponent_titles": opponent_titles,
            "player_runner_up_titles": player_runner_up_titles,
            "opponent_runner_up_titles": opponent_runner_up_titles,
        }


class MetricsCollector:
    def __init__(self, historical_games):
        self.events = []
        self.match_progress = []
        self.historical_games = historical_games
        self.hist_tree = self.build_tree_of_pong(historical_games)
        self.most_similar_game = None

    @staticmethod
    def scores_to_vec(scores, k):
        arr = np.full((k, 2), -1, dtype=np.int8)
        for i, s in enumerate(scores[:k]):
            l, r = map(int, s.split(":"))
            arr[i] = (l, r)
        return arr.ravel()

    def build_tree_of_pong(self, historical_games):
        logger.info("Indexing historical games.....")
        match_progress = historical_games["match_progress"].map(
            lambda x: self.scores_to_vec(x, GAME_POINT)
        )
        prefix_vectors = np.vstack(match_progress)
        return KDTree(prefix_vectors, metric="manhattan")

    def most_similar(self):
        if len(self.match_progress) == 0:
            return

        left, right = [int(x) for x in self.match_progress[-1].split(":")]
        if (
            max(left, right) <= GAME_POINT // 2
            or len(self.match_progress) < GAME_POINT // 1.5
        ):
            logger.info(
                "\033[91mNo similar game found, as the game is still in the early stages.\033[0m"
            )
            return

        query_vec = self.scores_to_vec(self.match_progress, GAME_POINT)
        dist, ind = self.hist_tree.query(query_vec.reshape(1, -1), k=1)
        row_ids = self.historical_games.index.to_numpy()
        similar_game = self.historical_games.iloc[row_ids[ind[0]]].assign(
            similarity=dist[0]
        )
        self.most_similar_game = similar_game

    def record_event(self, event_type, player=None, data=None):
        # Record the event with a timestamp
        # The event type can be one of the following:
        # - ball_bounce
        # - shot_speed
        # - serve_speed
        # - point_scored
        # - paddle_position
        # We use L and R to indicate left and right players
        # This is encoded in the player
        event = {
            "type": event_type,
            "player": player,
            "data": data,
            "timestamp": time.time(),
        }
        # Special case: we use this data to dig into the historical games
        if event_type == "point_scored":
            self.match_progress.append(data)
        # logger.info(f"Event recorded: {event}")
        self.events.append(event)

    def get_recent_events(self, past_seconds):
        if past_seconds == -1:
            return self.events
        cutoff = time.time() - past_seconds
        recent_events = [event for event in self.events if event["timestamp"] >= cutoff]
        return recent_events

    def get_score_metrics(self):
        left_score, right_score = 0, 0
        for event in self.events:
            if event["type"] == "point_scored":
                if event["player"] == "L":
                    left_score += 1
                elif event["player"] == "R":
                    right_score += 1
        return left_score, right_score

    def get_shot_speed_events(self, events):
        left_shot_speed, right_shot_speed = [], []
        for event in events:
            if event["type"] == "shot_speed":
                player = event["player"]
                speed = event["data"]
                if player == "L":
                    left_shot_speed.append(speed)
                elif player == "R":
                    right_shot_speed.append(speed)
        return left_shot_speed, right_shot_speed

    def get_serve_speed_events(self, events):
        left_serve_speed, right_serve_speed = [], []
        for event in events:
            if event["type"] == "serve_speed":
                player = event["player"]
                speed = event["data"]
                if player == "L":
                    left_serve_speed.append(speed)
                elif player == "R":
                    right_serve_speed.append(speed)
        return left_serve_speed, right_serve_speed

    def get_max_streaks(self):
        left_streak, right_streak = 0, 0
        left_max_streak, right_max_streak = 0, 0
        for event in self.events:
            if event["type"] == "point_scored":
                player = event["player"]
                if player == "L":
                    left_streak += 1
                    right_streak = 0
                    left_max_streak = max(left_max_streak, left_streak)
                elif player == "R":
                    right_streak += 1
                    left_streak = 0
                    right_max_streak = max(right_max_streak, right_streak)
        return left_max_streak, right_max_streak

    def get_recent_rally(self, events):
        rally_count = 0
        for event in events:
            if event["type"] in ("shot_speed", "serve_speed"):
                rally_count += 1
            elif event["type"] == "point_scored":
                rally_count = 0
        return rally_count

    def get_all_rally(self, events):
        rally_count = 0
        total_rallies = []
        for event in events:
            if event["type"] in ("shot_speed", "serve_speed"):
                rally_count += 1
            elif event["type"] == "point_scored":
                total_rallies.append(rally_count)
                rally_count = 0
        return total_rallies

    def convert_to_scalar_speed(self, data):
        if not data:
            return None
        game_speed = lambda vx, vy: round(math.sqrt(vx**2 + vy**2), 2)
        data = [game_speed(vx, vy) for vx, vy in data]
        # Deltas are added to prevent capping exactly at 160
        min_game_speed = 6.4  # approx sqrt(5**2 + 4**2)
        max_game_speed = 16.5  # approx sqrt(14**2 + 19**2)
        mph_min = 60
        mph_max = 100
        mph = (
            lambda speed: ((speed - min_game_speed) / (max_game_speed - min_game_speed))
            * (mph_max - mph_min)
            + mph_min
            # + random.uniform(-5, 5)  # to prevent players from getting similar numbers
        )
        return [round(mph(speed), 2) for speed in data]

    def shot_to_scalar_speed(self, last_shot_speed):
        if not last_shot_speed:
            return None
        return self.convert_to_scalar_speed([last_shot_speed])[0]

    def get_ball_bounce_count(self, events):
        ball_bounce_count = 0
        for event in events:
            if event["type"] == "ball_bounce":
                ball_bounce_count += 1
            elif event["type"] == "point_scored":
                ball_bounce_count = 0
        return ball_bounce_count

    def get_shot_direction_events(self, events):
        left_dirs, right_dirs = [], []
        for event in events:
            if event["type"] == "shot_direction":
                if event["player"] == "L":
                    left_dirs.append(event["data"])
                elif event["player"] == "R":
                    right_dirs.append(event["data"])
        return left_dirs, right_dirs

    def average_shot_angle(self, events):
        left_dirs, right_dirs = self.get_shot_direction_events(events)
        left_angles = (
            [math.degrees(math.atan2(vy, vx)) for vx, vy in left_dirs]
            if left_dirs
            else []
        )
        right_angles = (
            [math.degrees(math.atan2(vy, vx)) for vx, vy in right_dirs]
            if right_dirs
            else []
        )
        avg_left = sum(left_angles) / len(left_angles) if left_angles else None
        avg_right = sum(right_angles) / len(right_angles) if right_angles else None
        return avg_left, avg_right

    def get_average_paddle_movement(self, events):
        left_moves, right_moves = [], []
        last_left = last_right = None
        for event in events:
            if event["type"] == "paddle_position":
                curr_left = event["data"].get("L")
                curr_right = event["data"].get("R")
                if last_left is not None:
                    left_moves.append(abs(curr_left - last_left))
                if last_right is not None:
                    right_moves.append(abs(curr_right - last_right))
                last_left, last_right = curr_left, curr_right
        avg_left = sum(left_moves) / len(left_moves) if left_moves else None
        avg_right = sum(right_moves) / len(right_moves) if right_moves else None
        return avg_left, avg_right

    def compute_metrics(self, past_seconds):
        # Compute metrics based on the recorded events
        # These metrics are:
        # Per player:
        # - Score
        # - Recent most, average, minimum, and maximum speed from shots and serves
        # - Previous serve quality
        # - Recent most, and average ball trajectory from shots and serves
        # - Recent most, and average paddle position from shots and serves
        # - Current and max streaks (winning and losing)
        # Both players:
        # - Average, and maximum rally
        events = self.get_recent_events(past_seconds)
        left_score, right_score = self.get_score_metrics()

        most_similar_game = None
        if random.random() < 0.33:
            self.most_similar()
            # It is better to do this,
            # so that we do not have to distribute probability calls during commentary generation
            # This is to prevent a probability cascade in here and in the commentary gen code
            # For example, if this prob is 0.5, and the commentary gen prob is 0.5,
            # then we are looking at 0.25 chance of getting a historical commentary in
            # But it is better to tune this number right here
            most_similar_game = self.most_similar_game

        left_shot_speeds, right_shot_speeds = self.get_shot_speed_events(events)
        left_shot_speeds = self.convert_to_scalar_speed(left_shot_speeds)
        right_shot_speeds = self.convert_to_scalar_speed(right_shot_speeds)

        recent_left_shot_speed = (
            round(left_shot_speeds[-1]) if left_shot_speeds else None
        )
        recent_right_shot_speed = (
            round(right_shot_speeds[-1]) if right_shot_speeds else None
        )
        left_shot_speed_avg = (
            int(sum(left_shot_speeds) / len(left_shot_speeds))
            if left_shot_speeds
            else None
        )
        right_shot_speed_avg = (
            int(sum(right_shot_speeds) / len(right_shot_speeds))
            if right_shot_speeds
            else None
        )
        # left_shot_speed_max = max(left_shot_speeds) if left_shot_speeds else None
        # right_shot_speed_max = max(right_shot_speeds) if right_shot_speeds else None

        left_serve_speeds, right_serve_speeds = self.get_serve_speed_events(events)
        left_serve_speeds = self.convert_to_scalar_speed(left_serve_speeds)
        right_serve_speeds = self.convert_to_scalar_speed(right_serve_speeds)

        recent_left_serve_speed = (
            round(left_serve_speeds[-1]) if left_serve_speeds else None
        )
        recent_right_serve_speed = (
            round(right_serve_speeds[-1]) if right_serve_speeds else None
        )
        left_serve_speed_avg = (
            int(sum(left_serve_speeds) / len(left_serve_speeds))
            if left_serve_speeds
            else None
        )
        right_serve_speed_avg = (
            int(sum(right_serve_speeds) / len(right_serve_speeds))
            if right_serve_speeds
            else None
        )
        # left_serve_speed_max = max(left_serve_speeds) if left_serve_speeds else None
        # right_serve_speed_max = max(right_serve_speeds) if right_serve_speeds else None

        left_max_streak, right_max_streak = self.get_max_streaks()

        recent_rally = self.get_recent_rally(events)
        all_rallies = self.get_all_rally(events)
        average_rally = sum(all_rallies) / len(all_rallies) if all_rallies else None

        ball_bounce_count = self.get_ball_bounce_count(events)
        avg_left_shot_angle, avg_right_shot_angle = self.average_shot_angle(events)
        avg_left_paddle_movement, avg_right_paddle_movement = (
            self.get_average_paddle_movement(events)
        )

        return {
            "left_score": left_score,
            "right_score": right_score,
            "recent_left_shot_speed": recent_left_shot_speed,
            "recent_right_shot_speed": recent_right_shot_speed,
            "left_shot_speed_avg": left_shot_speed_avg,
            "right_shot_speed_avg": right_shot_speed_avg,
            "recent_left_serve_speed": recent_left_serve_speed,
            "recent_right_serve_speed": recent_right_serve_speed,
            "left_serve_speed_avg": left_serve_speed_avg,
            "right_serve_speed_avg": right_serve_speed_avg,
            "left_max_streak": left_max_streak,
            "right_max_streak": right_max_streak,
            "average_rally": average_rally,
            "recent_rally": recent_rally,
            "ball_bounce_count": ball_bounce_count,
            "avg_left_shot_angle": avg_left_shot_angle,
            "avg_right_shot_angle": avg_right_shot_angle,
            "avg_left_paddle_movement": avg_left_paddle_movement,
            "avg_right_paddle_movement": avg_right_paddle_movement,
            "most_similar_game": most_similar_game,
        }


class PongGame:
    def __init__(self, all_time_greats, historical_games):
        self.metrics = MetricsCollector(historical_games)

        self.width = 800
        self.height = 600

        self.ball = {
            "x": self.width / 2,
            # this is the center of the ball
            "y": self.height / 2,
            # speed and dir in x dir
            "vx": 5,
            # speed and dir in y dir
            "vy": 3,
            "radius": 10,
        }

        self.left_paddle = {
            "x": 10,
            # this is the top of the paddle
            "y": self.height / 2 - 50,
            "width": 10,
            "height": 100,
            "speed": 6,
        }
        self.right_paddle = {
            "x": self.width - 20,
            "y": self.height / 2 - 50,
            "width": 10,
            "height": 100,
            "speed": 6,
        }

        self.left_score = 0
        self.right_score = 0

        self.rally_count = 0
        self.ball_bounce_count = 0
        self.last_shot_player = None

        self.left_last_shot_speed = None
        self.right_last_shot_speed = None

        # Approximately 60 FPS
        self.game_speed = 0.016 / XFPS

        self.ball_in_play = True

        eel.init("./web")

        self.gpt_prompts = GPTPrompts(all_time_greats)
        self.h2h_stats = None
        self.last_commentary_time = None
        self.last_metrics_snapshot = None
        self.commentary_manager = CommentaryManager(self.gpt_prompts)

        self.paused = False
        self.game_over = False

    def init_ball(self, direction):
        # If left side just lost, we serve on left side
        if direction == 1:
            # Position ball next to the left paddle
            self.ball["x"] = (
                self.left_paddle["x"] + self.left_paddle["width"] + self.ball["radius"]
            )
            # Place the ball vertically at a random part of the left paddle
            self.ball["y"] = self.left_paddle["y"] + random.randrange(
                math.ceil(self.left_paddle["height"] * 0.25),
                math.ceil(self.left_paddle["height"] * 0.75),
            )
        # If right side just lost, we serve on right side
        elif direction == -1:
            self.ball["x"] = self.right_paddle["x"] - self.ball["radius"]
            self.ball["y"] = self.right_paddle["y"] + random.randrange(
                math.ceil(self.right_paddle["height"] * 0.25),
                math.ceil(self.right_paddle["height"] * 0.75),
            )

        self.shot_velocity_x(direction)
        self.play_paddle_shot_sound()
        self.shot_velocity_y()

    async def reset_ball(self, direction=1):
        if not SPEEDUP:
            await asyncio.sleep(8)
        self.init_ball(direction)
        self.ball_in_play = True

    def shot_velocity_x(self, direction):
        self.ball["vx"] = direction * random.choice([i for i in range(5, 15)])

    def shot_velocity_y(self):
        self.ball["vy"] = random.choice(
            [i for i in range(-4, -10, -1)] + [i for i in range(4, 10)]
        )

    def play_paddle_shot_sound(self):
        # variable volume between 0.1 and 1, based on the new shot speed
        # when the shot is softest, the velocity is 5, and volume is 0.1
        # when the shot is hardest, the velocity is 15, and volume is 1
        volume = ((abs(self.ball["vx"]) - 5) / 10) * 0.9 + 0.1
        eel.play_sound(volume)

    def predict_ball_y(self, paddle_x):
        net_line = self.width / 2
        # if paddle is on the left and ball is moving to the right,
        # stand at the center mark, and vice versa
        if (paddle_x < net_line and self.ball["vx"] >= 0) or (
            paddle_x > net_line and self.ball["vx"] <= 0
        ):
            return self.height / 2

        # reaction time
        # if the paddle is on left, and ball is moving to the left,
        # we start the prediction after ball crosses the the net line
        # and vice versa
        # tuning this, changes how strong the AI can predict the ball
        if (paddle_x < net_line and self.ball["x"] > net_line) or (
            paddle_x > net_line and self.ball["x"] < net_line
        ):
            return self.height / 2

        # how long it will take for the ball to reach the paddle
        time_to_reach = abs((paddle_x - self.ball["x"]) / self.ball["vx"])
        # estimate where the paddle should be,
        # by adding up the ball's current y position
        # with the y dist it will take to reach the paddle
        y = self.ball["y"] + self.ball["vy"] * time_to_reach

        # if the ball is out of bounds, we need to reflect it
        while y < 0 or y > self.height:
            if y < 0:
                y = -y
            elif y > self.height:
                y = 2 * self.height - y
        return y

    def move_paddle_toward(self, paddle, target_y):
        center = paddle["y"] + paddle["height"] / 2
        # paddle speed determines the accuracy of the prediction
        # and the speed of the paddle AI
        if abs(center - target_y) < paddle["speed"]:
            return
        if center < target_y:
            # if the paddle is not overshooting across the bottom of the screen,
            # move it closer to the predicted y position
            if paddle["y"] + paddle["height"] < self.height:
                paddle["y"] += paddle["speed"]
        elif center > target_y:
            # if the paddle is not overshooting across the top of the screen,
            # move it closer to the predicted y position
            if paddle["y"] > 0:
                paddle["y"] -= paddle["speed"]

    def update_game(self):
        if self.paused or self.game_over:
            return

        if not self.ball_in_play:
            return

        self.ball["x"] += self.ball["vx"]
        self.ball["y"] += self.ball["vy"]

        # collision with top and bottom walls
        if (
            self.ball["y"] - self.ball["radius"] <= 0
            or self.ball["y"] + self.ball["radius"] >= self.height
        ):
            self.metrics.record_event(event_type="ball_bounce")
            self.ball_bounce_count += 1
            self.ball["vy"] *= -1

        # AI paddle movement for left paddle
        # There is no fancy AI here.
        # We simply check where the ball is going to be in the y direction,
        # based on the ball's velocity,
        # and move the paddle to the predicted y position (within an approx range)
        left_target = (
            self.predict_ball_y(self.left_paddle["x"])
            if self.ball["vx"] < 0
            else self.height / 2
        )
        self.move_paddle_toward(self.left_paddle, left_target)

        # AI paddle movement for right paddle
        # Similar to the left paddle
        right_target = (
            self.predict_ball_y(self.right_paddle["x"])
            if self.ball["vx"] > 0
            else self.height / 2
        )
        self.move_paddle_toward(self.right_paddle, right_target)

        # if the ball hits the left paddle
        # we need the left paddle to make a new shot with a new velocity
        if (
            self.ball["x"] - self.ball["radius"]
            <= self.left_paddle["x"] + self.left_paddle["width"]
            and self.left_paddle["y"]
            <= self.ball["y"]
            <= self.left_paddle["y"] + self.left_paddle["height"]
            and self.last_shot_player != "L"
        ):
            self.last_shot_player = "L"
            self.shot_velocity_x(direction=1)
            self.shot_velocity_y()
            self.left_last_shot_speed = abs(self.ball["vx"]), abs(self.ball["vy"])
            self.rally_count += 1
            self.metrics.record_event(
                event_type="shot_speed",
                player="L",
                data=(self.left_last_shot_speed),
            )
            self.metrics.record_event(
                event_type="paddle_position",
                player="L",
                data={
                    "L": self.left_paddle["y"],
                    "R": self.right_paddle["y"],
                },
            )
            self.metrics.record_event(
                event_type="shot_direction",
                player="L",
                data=(self.ball["vx"], self.ball["vy"]),
            )
            self.play_paddle_shot_sound()

        # ball hitting the right paddle
        # similar to the left paddle but in the opposite direction
        if (
            self.ball["x"] + self.ball["radius"] >= self.right_paddle["x"]
            and self.right_paddle["y"]
            <= self.ball["y"]
            <= self.right_paddle["y"] + self.right_paddle["height"]
            and self.last_shot_player != "R"
        ):
            self.last_shot_player = "R"
            self.shot_velocity_x(direction=-1)
            self.shot_velocity_y()
            self.right_last_shot_speed = abs(self.ball["vx"]), abs(self.ball["vy"])
            self.rally_count += 1
            self.metrics.record_event(
                event_type="shot_speed",
                player="R",
                data=(self.right_last_shot_speed),
            )
            self.metrics.record_event(
                event_type="paddle_position",
                player="R",
                data={
                    "L": self.left_paddle["y"],
                    "R": self.right_paddle["y"],
                },
            )
            self.metrics.record_event(
                event_type="shot_direction",
                player="R",
                data=(self.ball["vx"], self.ball["vy"]),
            )
            self.play_paddle_shot_sound()

        # ball's out of bounds
        if self.ball["x"] < 0:
            self.ball_in_play = False
            self.right_score += 1
            self.rally_count = 1  # to account for the serve
            self.ball_bounce_count = 0
            self.last_shot_player = None
            match_progress = f"{self.left_score}:{self.right_score}"
            self.metrics.record_event(
                event_type="point_scored", player="R", data=match_progress
            )
            if self.right_score >= GAME_POINT:
                self.game_over = True
                asyncio.create_task(self.end_game())
                return
            asyncio.create_task(self.reset_ball(direction=1))
            self.metrics.record_event(
                event_type="serve_speed",
                player="L",
                data=(abs(self.ball["vx"]), abs(self.ball["vy"])),
            )
            self.commentary_manager.flush()
            self.last_commentary_time = (
                time.time()
            )  # prevents double commentary in the game loop
            asyncio.create_task(
                self.generate_and_enqueue_commentary(score_change=True, scored_by="R"),
            )
        elif self.ball["x"] > self.width:
            self.ball_in_play = False
            self.left_score += 1
            self.rally_count = 1
            self.ball_bounce_count = 0
            self.last_shot_player = None
            match_progress = f"{self.left_score}:{self.right_score}"
            self.metrics.record_event(
                event_type="point_scored", player="L", data=match_progress
            )
            if self.left_score >= GAME_POINT:
                self.game_over = True
                asyncio.create_task(self.end_game())
                return
            asyncio.create_task(self.reset_ball(direction=-1))
            self.metrics.record_event(
                event_type="serve_speed",
                player="R",
                data=(abs(self.ball["vx"]), abs(self.ball["vy"])),
            )
            self.commentary_manager.flush()
            self.last_commentary_time = time.time()
            asyncio.create_task(
                self.generate_and_enqueue_commentary(score_change=True, scored_by="L"),
            )

    async def generate_and_enqueue_commentary(self, score_change=False, scored_by=None):
        if NO_COMMENTARY:
            return

        base_metrics = await asyncio.to_thread(self.metrics.compute_metrics, -1)

        base_metrics["left_player_name"] = self.h2h_stats["player_name"]
        base_metrics["right_player_name"] = self.h2h_stats["opponent_name"]
        base_metrics["player_titles"] = self.h2h_stats["player_titles"]
        base_metrics["opponent_titles"] = self.h2h_stats["opponent_titles"]
        base_metrics["player_runner_up_titles"] = self.h2h_stats[
            "player_runner_up_titles"
        ]
        base_metrics["opponent_runner_up_titles"] = self.h2h_stats[
            "opponent_runner_up_titles"
        ]
        base_metrics["left_player_style"] = self.h2h_stats["player_style"]
        base_metrics["right_player_style"] = self.h2h_stats["opponent_style"]
        base_metrics["left_player_country"] = self.h2h_stats["player_country"]
        base_metrics["right_player_country"] = self.h2h_stats["opponent_country"]
        base_metrics["left_player_world_rank"] = self.h2h_stats["player_rank"]
        base_metrics["right_player_world_rank"] = self.h2h_stats["opponent_rank"]

        scored_by = (
            base_metrics["left_player_name"]
            if scored_by == "L"
            else (base_metrics["right_player_name"] if scored_by == "R" else None)
        )
        commentary_script = await asyncio.to_thread(
            self.gpt_prompts.generate_in_game_commentary,
            base_metrics,
            self.commentary_manager.commentary_history,
            score_change,
            scored_by,
        )
        self.commentary_manager.enqueue(commentary_script)

    async def game_loop(self):
        NO_OPENING_SCRIPT = "--no-opening" in sys.argv
        if not NO_OPENING_SCRIPT:
            eel.show_match_card(self.h2h_stats)
            logger.info(f"Start the opening commentary script...")
            await self.gpt_prompts.speak_opening_script(self.h2h_stats)

        while True:
            if self.paused or self.game_over:
                # minimal ticking
                await asyncio.sleep(self.game_speed)
                continue

            self.update_game()
            # invoke index.html's update_pong function
            eel.update_pong(
                self.ball,
                self.left_paddle,
                self.right_paddle,
                self.left_score,
                self.right_score,
                self.width,
                self.height,
                self.h2h_stats["player_name"],
                self.h2h_stats["opponent_name"],
                self.h2h_stats["player_rank"],
                self.h2h_stats["opponent_rank"],
                self.h2h_stats["player_country"],
                self.h2h_stats["opponent_country"],
                self.metrics.shot_to_scalar_speed(self.left_last_shot_speed),
                self.metrics.shot_to_scalar_speed(self.right_last_shot_speed),
                self.rally_count,
                self.ball_bounce_count,
            )
            current_time = time.time()
            if (
                self.last_commentary_time is None
                or current_time - self.last_commentary_time > 20
            ):
                self.last_commentary_time = current_time
                asyncio.create_task(self.generate_and_enqueue_commentary())
            # wait for the next frame
            await asyncio.sleep(self.game_speed)

    def start_game(self, head_to_head_stats):
        self.h2h_stats = head_to_head_stats

        eel.expose(self.toggle_pause)

        def start_eel():
            eel.start(
                "index.html",
                size=(self.width, self.height),
                block=True,
                close_callback=self.close_window,
            )

        eel_thread = threading.Thread(target=start_eel)
        eel_thread.daemon = True
        eel_thread.start()

        time.sleep(1)
        self.init_ball(direction=random.choice([-1, 1]))
        asyncio.run(self.game_loop())

    def close_window(self, _route, _websockets):
        logger.info("Closing the game window...")
        # print(self.metrics.compute_metrics(past_seconds=-1))
        os._exit(0)

    def get_final_stats(self):
        if self.left_score == GAME_POINT:
            winner = self.h2h_stats["player_name"]
            loser = self.h2h_stats["opponent_name"]
        else:
            winner = self.h2h_stats["opponent_name"]
            loser = self.h2h_stats["player_name"]
        all_metrics = self.metrics.compute_metrics(past_seconds=-1)
        final_metrics = {}
        for key in all_metrics:
            # we do not want to show any recent metrics
            if "recent" not in key:
                final_metrics[key] = all_metrics[key]
        return (winner, loser, final_metrics)

    async def end_game(self):
        self.paused = True
        self.ball_in_play = False
        self.commentary_manager.flush(no_filler=True)

        logger.info(f"Start the closing commentary script...")
        winner, loser, final_metrics = self.get_final_stats()
        await self.gpt_prompts.speak_final_commentary(winner, loser, final_metrics)

        logger.info(
            f"Game Over! {winner} wins with score {self.left_score}:{self.right_score}"
        )
        self.close_window(None, None)

    def toggle_pause(self):
        # toggle pause
        self.paused = not self.paused
        if self.paused:
            self.commentary_manager.flush(no_filler=True)
        self.last_commentary_time = time.time()


if __name__ == "__main__":
    CACHE_ENABLED = "--cache" in sys.argv
    PLAYER_INFO_CACHE = "./assets/player_info.pkl"

    logger.info("Generating tournament data...")
    simul = GameStats(num_top_players=64)
    gpt_prompts = GPTPrompts()
    logger.info(f"{simul.num_top_players} top players in the world...")

    player_ids = list(range(1, simul.num_top_players + 1))
    player_rankings = list(range(1, simul.num_top_players + 1))

    if CACHE_ENABLED and os.path.exists(PLAYER_INFO_CACHE):
        logger.info("Loading player information from cache...")
        with open(PLAYER_INFO_CACHE, "rb") as f:
            player_info = pickle.load(f)
    else:
        logger.info("Generating player information from GPT...")
        player_info = gpt_prompts.generate_player_info()
        logger.info("Caching player information...")
        with open(PLAYER_INFO_CACHE, "wb") as f:
            pickle.dump(player_info, f)

    logger.info("Shuffle player rankings and assign ELO ratings...")
    random.shuffle(player_rankings)
    simul.assign_init_elo(player_ids, player_rankings)

    num_tournament_yrs = 15
    tournament_year = 2025 - num_tournament_yrs
    logger.info(
        f"Simulating {num_tournament_yrs} tournament years starting with year {tournament_year}..."
    )
    for season in range(0, num_tournament_yrs):
        for i, tournament in enumerate(simul.major_tournaments):
            tournament_id = season + i + (3 * season)
            simul.simulate_tournament(
                player_ids, tournament, tournament_id, tournament_year
            )
        tournament_year += 1
    # At this moment, only game and tournament stats are simulated
    logger.info(f"Storing historical games and tournament data.....")
    simulation_dir = "./simulation"
    os.makedirs(f"{simulation_dir}", exist_ok=True)
    simul.game_stats.to_csv(f"{simulation_dir}/historical_games.csv", index=False)
    simul.tournament_stats.to_csv(
        f"{simulation_dir}/tournament_results.csv", index=False
    )

    logger.info(f"Generate player statistics based on tournament data...")
    sorted_players = sorted(simul.player_elo.items(), key=lambda x: x[1], reverse=True)
    simul.player_statistics(player_ids, player_info)
    simul.player_stats.to_csv(f"{simulation_dir}/players.csv", index=False)

    logger.info(f"Top and bottom ELO statistics...")
    simul.show_top_bottom_elo_stats(sorted_players)

    logger.info(f"Generate head-to-head statistics for the top 2 players...")
    head_to_head_stats = simul.head_to_head_statistics(
        sorted_players[0][0], sorted_players[1][0]
    )

    logger.info(f"Fetching all-time greats...")
    simul.get_all_time_greats()

    logger.info("Starting Pong Game...")
    PongGame(simul.all_time_greats, simul.game_stats).start_game(head_to_head_stats)

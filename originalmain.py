import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import requests  # For GPT-4o Mini API requests
import json
import random  # For random sampling
import threading

# --------------------------- Configuration --------------------------- #

# Spotify API credentials
SPOTIPY_CLIENT_ID = 'fe6a043531214ac1abf575878699f986'
SPOTIPY_CLIENT_SECRET = '3ca2ef78541a47109b884f42a90881db'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# GPT-4o Mini API credentials and endpoint
GPT4_MINI_API_KEY = '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'  # Replace with your actual API key
GPT4_MINI_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'

# --------------------------- Initialization --------------------------- #

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope='playlist-read-private playlist-read-collaborative playlist-modify-public playlist-modify-private'
), requests_timeout=10)  # Set a timeout for API requests

# --------------------------- Helper Functions --------------------------- #

def call_gpt4_mini(prompt):
    """
    Calls the GPT-4o Mini API with the provided prompt and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {GPT4_MINI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(GPT4_MINI_API_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()  # Raises HTTPError for bad responses
        result = response.json()
        if 'choices' not in result or not result['choices']:
            print("No choices in GPT-4o Mini response.")
            print("Response:", result)
            return None
        # Extract the assistant's message content
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # HTTP error
        print(f"Response Text: {response.text}")  # Debugging line
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")  # JSON error
        print(f"Response Text: {response.text}")  # Print raw response
    except Exception as err:
        print(f"Other error occurred: {err}")  # Other errors
        if 'response' in locals():
            print(f"Response Text: {response.text}")  # Print raw response
    return None

def get_user_input():
    """
    Collects user inputs for genre, mood, energy level, optional additional info, and optional playlists.
    """
    genre = input("Enter your preferred genre (e.g., Rock, Pop): ").strip()
    mood = input("Enter desired mood (e.g., Calm, Happy, Sad): ").strip()
    energy_level = input("Select energy level (Low, Medium, High): ").strip().capitalize()
    
    # Optional additional info
    additional_info = input("Any additional information or preferences? (optional): ").strip()
    
    # Optional: Ask if user wants to include existing playlists
    include_playlists = input("Do you want to include an existing playlist for analysis? (yes/no): ").strip().lower()
    playlists = []
    if include_playlists == 'yes':
        user_playlists = sp.current_user_playlists(limit=50)
        print("\nYour Playlists:")
        for idx, playlist in enumerate(user_playlists['items'], 1):
            print(f"{idx}. {playlist['name']} (ID: {playlist['id']})")
        selected = input("Enter the playlist numbers to include, separated by commas (e.g., 1,3,5): ").strip()
        selected_indices = [int(x) - 1 for x in selected.split(',') if x.isdigit()]
        for idx in selected_indices:
            if 0 <= idx < len(user_playlists['items']):
                playlists.append(user_playlists['items'][idx]['id'])
    return genre, mood, energy_level, additional_info, playlists

def determine_search_parameters(genre, mood, energy_level, additional_info):
    """
    Uses GPT-4o Mini to determine which genres and artists to search for based on user inputs.
    """
    prompt = f"""
Based on the following user preferences:
- Genre: {genre}
- Mood: {mood}
- Energy Level: {energy_level}
- Additional Info: {additional_info if additional_info else 'None'}

Please provide a list of genres and artists that match these preferences.

**Respond ONLY with valid JSON in the following format, without any code fences or additional text:**

{{
    "genres": ["genre1", "genre2", ...],
    "artists": ["artist1", "artist2", ...]
}}
"""
    
    response = call_gpt4_mini(prompt)
    if response:
        try:
            # Strip leading and trailing whitespace
            response = response.strip()
            # Remove code fences if present
            if response.startswith("```") and response.endswith("```"):
                # Remove the starting and ending triple backticks
                response = response[3:-3].strip()
                # Remove optional language identifier after the opening backticks
                if response.startswith('json'):
                    response = response[4:].strip()
            search_params = json.loads(response)
            genres = search_params.get('genres', [])
            artists = search_params.get('artists', [])
            return genres, artists
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing GPT-4o Mini response for search parameters: {e}")
            print("Raw GPT-4o Mini response:", response)
            return [genre], []
    else:
        return [genre], []

def search_and_combine_playlists(search_genres, search_artists, existing_playlist_ids, limit=200, sample_size_per_playlist=10):
    """
    Searches for playlists based on genres and artists, combines them with existing playlists.
    Ensures that the same playlist is not processed more than once.
    Returns a list of unique track IDs limited to a specified count.
    """
    track_ids = set()
    processed_playlists = set()  # Set to keep track of processed playlist IDs
    
    print("Searching for playlists based on genres...")
    # Search for playlists matching the genres
    for genre in search_genres:
        print(f"Searching for playlists with genre: {genre}")
        search_results = sp.search(q=f'genre:"{genre}"', type='playlist', limit=10)
        playlist_items = search_results['playlists']['items']
        if not playlist_items:
            print(f"No playlists found for genre '{genre}'.")
            continue
        selected_playlists = playlist_items  # Use all found playlists

        for playlist in selected_playlists:
            playlist_id = playlist['id']
            # Check if playlist has already been processed
            if playlist_id in processed_playlists:
                print(f"Skipping already processed playlist '{playlist['name']}' (ID: {playlist_id}).")
                continue
            # Get the total number of tracks in the playlist
            playlist_size = playlist['tracks']['total']
            # Check playlist size against the specified criteria
            if playlist_size < 100 or playlist_size > 1500:
                print(f"Skipping playlist '{playlist['name']}' (has {playlist_size} tracks).")
                continue
            print(f"Fetching tracks from playlist: {playlist['name']}")
            fetched_track_ids = fetch_playlist_tracks(playlist_id, sample_size=sample_size_per_playlist)
            print(f"Fetched {len(fetched_track_ids)} tracks from playlist: {playlist['name']}")
            track_ids.update(fetched_track_ids)
            processed_playlists.add(playlist_id)  # Mark this playlist as processed

    print("Searching for playlists based on artists...")
    # Search for playlists matching the artists
    for artist in search_artists:
        print(f"Searching for playlists with artist: {artist}")
        search_results = sp.search(q=f'artist:"{artist}"', type='playlist', limit=10)
        playlist_items = search_results['playlists']['items']
        if not playlist_items:
            print(f"No playlists found for artist '{artist}'.")
            continue
        selected_playlists = playlist_items  # Use all found playlists

        for playlist in selected_playlists:
            playlist_id = playlist['id']
            # Check if playlist has already been processed
            if playlist_id in processed_playlists:
                print(f"Skipping already processed playlist '{playlist['name']}' (ID: {playlist_id}).")
                continue
            # Get the total number of tracks in the playlist
            playlist_size = playlist['tracks']['total']
            # Check playlist size against the specified criteria
            if playlist_size < 100 or playlist_size > 1500:
                print(f"Skipping playlist '{playlist['name']}' (has {playlist_size} tracks).")
                continue
            print(f"Fetching tracks from playlist: {playlist['name']}")
            fetched_track_ids = fetch_playlist_tracks(playlist_id, sample_size=sample_size_per_playlist)
            print(f"Fetched {len(fetched_track_ids)} tracks from playlist: {playlist['name']}")
            track_ids.update(fetched_track_ids)
            processed_playlists.add(playlist_id)  # Mark this playlist as processed

    # Include tracks from existing playlists
    if existing_playlist_ids:
        print("Including tracks from existing playlists...")
        for playlist_id in existing_playlist_ids:
            # Check if playlist has already been processed
            if playlist_id in processed_playlists:
                print(f"Skipping already processed existing playlist (ID: {playlist_id}).")
                continue
            # Get playlist size and name
            playlist = sp.playlist(playlist_id, fields='tracks.total,name')
            playlist_size = playlist['tracks']['total']
            playlist_name = playlist['name']
            if playlist_size < 100 or playlist_size > 1500:
                print(f"Skipping existing playlist '{playlist_name}' (has {playlist_size} tracks).")
                continue
            print(f"Fetching tracks from existing playlist: {playlist_name}")
            fetched_track_ids = fetch_playlist_tracks(playlist_id, sample_size=sample_size_per_playlist)
            print(f"Fetched {len(fetched_track_ids)} tracks from existing playlist: {playlist_name}")
            track_ids.update(fetched_track_ids)
            processed_playlists.add(playlist_id)  # Mark this playlist as processed

    # Return only up to the specified limit
    track_ids = list(track_ids)
    if len(track_ids) > limit:
        track_ids = random.sample(track_ids, limit)
    return track_ids

def fetch_playlist_tracks(playlist_id, sample_size=10):
    """
    Efficiently fetches all track IDs from a given playlist and selects a random sample.
    """
    track_ids = []
    try:
        # Get the total number of tracks in the playlist
        total_tracks = sp.playlist_items(playlist_id, fields='total')['total']
        print(f"Total tracks in playlist: {total_tracks}")

        # Fetch all track IDs using pagination
        batch_size = 100  # Maximum allowed by Spotify
        for offset in range(0, total_tracks, batch_size):
            results = sp.playlist_items(
                playlist_id,
                fields='items.track.id',
                additional_types=['track'],
                limit=batch_size,
                offset=offset
            )
            items = results['items']
            for item in items:
                track = item['track']
                if track and track['id']:
                    track_ids.append(track['id'])
        
        # Randomly select sample_size track IDs
        if len(track_ids) >= sample_size:
            track_ids = random.sample(track_ids, sample_size)
        else:
            print(f"Playlist {playlist_id} has fewer tracks than the sample size. Using all available tracks.")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error fetching playlist {playlist_id}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while fetching playlist {playlist_id}: {e}")
    return track_ids

def get_audio_features_concurrent(track_ids):
    """
    Retrieves audio features for a list of track IDs using threading for concurrency.
    """
    audio_features = []
    lock = threading.Lock()
    
    def fetch_batch(start):
        batch = track_ids[start:start + 100]
        try:
            features = sp.audio_features(batch)
            with lock:
                audio_features.extend([f for f in features if f is not None])
        except Exception as e:
            print(f"Error fetching audio features for batch starting at index {start}: {e}")
    
    threads = []
    for i in range(0, len(track_ids), 100):
        thread = threading.Thread(target=fetch_batch, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return audio_features

def get_track_info_concurrent(track_ids):
    """
    Retrieves track information (name, artists) for a list of track IDs using threading for concurrency.
    """
    track_info_list = []
    lock = threading.Lock()
    
    def fetch_batch(start):
        batch = track_ids[start:start+50]
        try:
            tracks = sp.tracks(batch)['tracks']
            with lock:
                for track in tracks:
                    if track and track['id']:
                        track_info = {
                            'id': track['id'],
                            'name': track['name'],
                            'artists': ', '.join([artist['name'] for artist in track['artists']])
                        }
                        track_info_list.append(track_info)
        except Exception as e:
            print(f"Error fetching track info for batch starting at index {start}: {e}")
    
    threads = []
    for i in range(0, len(track_ids), 50):
        thread = threading.Thread(target=fetch_batch, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return track_info_list

def filter_songs_with_model(df, diagnostic, feature_definitions):
    """
    Filters songs using GPT-4o Mini based on the song features and the user's diagnostic.
    """
    filtered_ids = []
    
    batch_size = 7  # Adjusted batch size as per your request
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
    
        # Construct prompt
        prompt = construct_prompt(batch_df, diagnostic, feature_definitions)
    
        # Call GPT-4o Mini
        response = call_gpt4_mini(prompt)
    
        if response is None:
            continue  # Skip this batch if there's an error
    
        # Parse response
        ids_to_keep = parse_model_response(response)
        filtered_ids.extend(ids_to_keep)
    
    return filtered_ids

def construct_prompt(batch_df, diagnostic, feature_definitions):
    """
    Constructs a prompt for GPT-4o Mini with the song data, the diagnostic, and feature definitions.
    """
    prompt = f"Based on the following user preferences:\n\n'{diagnostic}'\n\n"
    prompt += "Here are the definitions of the audio features:\n\n"
    prompt += feature_definitions + "\n\n"
    prompt += (
        "Please analyze the following songs and decide whether each song should be included in the playlist. "
        "For each song, provide a decision 'Keep' or 'Discard'. Here are the songs:\n\n"
    )

    songs_list = []
    for index, row in batch_df.iterrows():
        song_info = {
            "id": row["id"],
            "name": row["name"],
            "artists": row["artists"],
            "features": {
                "danceability": row["danceability"],
                "energy": row["energy"],
                "valence": row["valence"],
                "tempo": row["tempo"],
                "acousticness": row["acousticness"],
                "instrumentalness": row["instrumentalness"],
                "liveness": row["liveness"],
                "loudness": row["loudness"],
                "speechiness": row["speechiness"],
            },
        }
        songs_list.append(song_info)

    # Convert songs_list to JSON with double quotes
    songs_json = json.dumps(songs_list, indent=2)
    prompt += songs_json + "\n\n"

    prompt += (
        "Respond ONLY with a JSON array of decisions in the following format, "
        "ensuring all property names and string values are enclosed in double quotes, "
        "and without any code fences, explanations, or additional text:\n\n"
        '[\n  {"id": "song_id", "decision": "Keep" or "Discard"},\n  ...\n]\n'
    )

    return prompt

def parse_model_response(response):
    """
    Parses the GPT-4o Mini response to extract song IDs to keep.
    """
    try:
        response = response.strip()
        # Remove code fences if present
        if response.startswith("```") and response.endswith("```"):
            # Remove the starting and ending triple backticks
            response = response[3:-3].strip()
            # Remove optional language identifier after the opening backticks
            if response.startswith('json'):
                response = response[4:].strip()
        decisions = json.loads(response)
        ids_to_keep = [item['id'] for item in decisions if item['decision'].lower() == 'keep']
        return ids_to_keep
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Failed to parse GPT-4o Mini response: {e}")
        print("Raw GPT-4o Mini response:", response)
        return []

def create_new_playlist(name, track_ids):
    """
    Creates a new Spotify playlist with the specified track IDs.
    """
    user_id = sp.current_user()['id']
    playlist = sp.user_playlist_create(user=user_id, name=name, public=True, description='Generated by Automated Playlist Generator')
    # Spotify API allows adding up to 100 tracks per request
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        sp.playlist_add_items(playlist_id=playlist['id'], items=batch)
    return playlist['external_urls']['spotify']

def get_feature_definitions():
    """
    Returns a string containing the definitions of Spotify audio features.
    """
    feature_definitions = """
- **acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- **danceability**: Describes how suitable a track is for dancing based on tempo, rhythm stability, beat strength, and overall regularity. 0.0 is least danceable, 1.0 is most danceable.
- **energy**: Measure from 0.0 to 1.0 representing intensity and activity. High energy tracks feel fast and loud.
- **instrumentalness**: Predicts whether a track contains no vocals. Values closer to 1.0 indicate a higher likelihood of no vocal content.
- **liveness**: Detects the presence of an audience in the recording. Values above 0.8 indicate a high probability of live performance.
- **loudness**: Overall loudness of a track in decibels (dB). Values typically range between -60 and 0 dB.
- **speechiness**: Detects the presence of spoken words. Values above 0.66 indicate tracks made entirely of spoken words.
- **valence**: Measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. High valence sounds more positive.
- **tempo**: The overall estimated tempo of a track in beats per minute (BPM).
"""
    return feature_definitions

# --------------------------- Main Functionality --------------------------- #

def main():
    # Step 1: Collect User Inputs
    genre, mood, energy_level, additional_info, existing_playlists = get_user_input()

    # Construct diagnostic from user inputs
    diagnostic = f"A playlist with {energy_level.lower()} energy, {mood.lower()} mood, in the {genre} genre."
    if additional_info:
        diagnostic += f" Additional info: {additional_info}"

    # Step 2: Determine Search Parameters using GPT-4o Mini
    search_genres, search_artists = determine_search_parameters(genre, mood, energy_level, additional_info)
    if not search_genres and not search_artists:
        print("No genres or artists suggested by GPT-4o Mini for searching.")
        return

    print(f"\nGenres to search: {search_genres}")
    print(f"Artists to search: {search_artists}")

    # Step 3: Search and Combine Playlists with a limit
    print("\nStarting search and combination of playlists...")
    combined_track_ids = search_and_combine_playlists(search_genres, search_artists, existing_playlists, limit=200, sample_size_per_playlist=10)
    print(f"\nTotal unique tracks collected (limited to 200): {len(combined_track_ids)}")

    if not combined_track_ids:
        print("No tracks found based on the specified genres and artists.")
        return

    # Step 4: Retrieve Audio Features
    print("\nRetrieving audio features for collected tracks...")
    audio_features = get_audio_features_concurrent(combined_track_ids)
    print(f"Retrieved audio features for {len(audio_features)} tracks.")

    # Step 5: Retrieve Track Info
    print("\nRetrieving track information...")
    track_info_list = get_track_info_concurrent(combined_track_ids)
    print(f"Retrieved track information for {len(track_info_list)} tracks.")

    # Step 6: Merge DataFrames
    print("\nMerging track information and audio features...")
    df_features = pd.DataFrame(audio_features)
    df_track_info = pd.DataFrame(track_info_list)
    df = pd.merge(df_track_info, df_features, on='id')
    print(f"Merged data contains {len(df)} tracks.")

    # Handle missing features by setting to NaN
    features_to_handle = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
                          'speechiness', 'valence', 'tempo', 'loudness']
    for feature in features_to_handle:
        if feature in df.columns:
            df[feature] = df[feature].astype(float)  # Ensure correct data type
            df[feature] = df[feature].fillna(float('nan'))

    # Step 7: Apply Filtering with GPT-4o Mini
    print("\nApplying filtering with GPT-4o Mini...")
    feature_definitions = get_feature_definitions()
    filtered_ids = filter_songs_with_model(df, diagnostic, feature_definitions)
    print(f"\nTracks after filtering: {len(filtered_ids)}")

    if not filtered_ids:
        print("No tracks match the specified criteria.")
        return

    # Step 8: Create New Playlist
    print("\nCreating new playlist...")
    playlist_name = f"{genre} - {mood} - {energy_level} Energy"
    new_playlist_url = create_new_playlist(playlist_name, filtered_ids)
    print(f"\nNew Playlist Created: {new_playlist_url}")

if __name__ == "__main__":
    main()
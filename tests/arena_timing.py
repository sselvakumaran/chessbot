from arena import Arena
from bot_base import load_bot

bot0 = load_bot("v2.1_basic_nn_leakyrelu.json")
bot1 = load_bot("v1.1_heuristic_pst.json")

arena0 = Arena(
  white=bot0,
  black=bot1,
  max_plies=200,
  log_moves=True
)
arena1= Arena(
  white=bot1,
  black=bot0,
  max_plies=200,
  log_moves=True,
)
print("WHITE: NN, BLACK: HEURISTIC")
arena0.play(n_games=10, verbose=True)
print("WHITE: HEURISTIC, BLACK: NN")
arena1.play(n_games=10, verbose=True)
# resources/environments (local copy)

`maze_cuboids.json` is a copy of `vamp/resources/environments/maze_cuboids.json`,
duplicated here (it's a small static file, unlike the FR3 mesh assets) so this
project is self-contained and buildable in Docker without a second build
context reaching into `vamp/`. Re-copy it if the maze definition changes
upstream.

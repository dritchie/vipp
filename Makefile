# AD stuff
ADFILES = $(wildcard src/*.adjs)
MACROS = src/ad/macros.js
TRANSFORM = src/ad/transform
ADFILES_TRANSFORMED = $(ADFILES:.adjs=.js)

# Browser stuff
SRCFILES = $(wildcard src/*.js)
BROWSERIFIED = vipp_browser.js
MINIFIED = vipp_browser.min.js
MAINFILE = src/main.js

all: $(ADFILES_TRANSFORMED)

src/%.js: src/%.adjs $(MACROS)
	$(TRANSFORM) $< > $@

browser: $(MINIFIED)

$(MINIFIED): $(BROWSERIFIED)
	uglifyjs $< -b ascii_only=true,beautify=false > $@

$(BROWSERIFIED): $(ADFILES_TRANSFORMED) $(SRCFILES)
	browserify -t brfs $(MAINFILE) > $@

clean:
	rm -f $(ADFILES_TRANSFORMED)
	rm -f $(BROWSERIFIED) $(MINIFIED)


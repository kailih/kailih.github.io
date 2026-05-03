require "active_support/all"
require 'nokogiri'
require 'open-uri'

module Helpers
  extend ActiveSupport::NumberHelper unless method_defined?(:number_to_human)
end

module Jekyll
  class GoogleScholarTotalCitationsTag < Liquid::Tag
    Cache = {}

    def initialize(tag_name, params, tokens)
      super
      @scholar_id = params.strip
    end

    def render(context)
      scholar_id = context[@scholar_id]

      return Cache[scholar_id] if Cache[scholar_id]

      profile_url = "https://scholar.google.com/citations?user=#{scholar_id}&hl=en"

      begin
        sleep(rand(1.5..3.5))
        doc = Nokogiri::HTML(URI.open(profile_url, "User-Agent" => "Ruby/#{RUBY_VERSION}"))

        # Total citations are in the first gsc_rsb_std cell in the stats table
        stats_cells = doc.css('td.gsc_rsb_std')
        citation_count = stats_cells[0]&.text&.gsub(',', '')&.to_i || 0

        result = Helpers.number_to_human(
          citation_count,
          format: '%n%u',
          precision: 2,
          units: { thousand: 'K', million: 'M', billion: 'B' }
        )
      rescue => e
        puts "Error fetching total Scholar citations for #{scholar_id}: #{e.class} - #{e.message}"
        result = "N/A"
      end

      Cache[scholar_id] = result
      result
    end
  end
end

Liquid::Template.register_tag('google_scholar_total_citations', Jekyll::GoogleScholarTotalCitationsTag)
